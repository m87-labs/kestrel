use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// A wrapper around a raw slice pointer that can be sent across threads.
/// SAFETY: The caller must ensure the underlying data remains valid and unmodified
/// for the lifetime of this struct.
struct SendableSlice {
    ptr: *const u8,
    len: usize,
}

// SAFETY: We only read from the slice and ensure the backing buffer stays alive.
unsafe impl Send for SendableSlice {}
unsafe impl Sync for SendableSlice {}

impl SendableSlice {
    fn as_slice(&self) -> &[u8] {
        // SAFETY: Caller guarantees the data is valid for the lifetime of this struct.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Core decode logic, operates on a raw byte slice.
fn decode_rgb(bytes: &[u8]) -> Result<Option<(u32, u32, Vec<u8>)>, String> {
    let img = match image::load_from_memory(bytes) {
        Ok(img) => img,
        Err(image::ImageError::Unsupported(_)) => return Ok(None),
        Err(e) => return Err(format!("Failed to decode image: {}", e)),
    };

    let rgb = img.into_rgb8();
    let (width, height) = rgb.dimensions();
    let raw = rgb.into_raw();
    Ok(Some((width, height, raw)))
}

/// Decode image bytes into a numpy array (H, W, C) in RGB format.
///
/// Returns None if the format is not supported (caller should fall back to pyvips).
/// This function releases the GIL during decoding for parallel performance.
#[pyfunction]
fn decode_image<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyBytes>,
) -> PyResult<Option<Bound<'py, PyArray3<u8>>>> {
    // Get raw pointer to bytes buffer - this is stable for the lifetime of the PyBytes object
    let bytes = data.as_bytes();

    // SAFETY: The PyBytes object is kept alive by the `data` reference, and we don't
    // release it until after allow_threads returns. The buffer is immutable.
    let sendable = SendableSlice {
        ptr: bytes.as_ptr(),
        len: bytes.len(),
    };

    // Release GIL during CPU-intensive decode
    let result = py.allow_threads(move || decode_rgb(sendable.as_slice()));

    match result {
        Ok(Some((width, height, raw))) => {
            let array = ndarray::Array3::from_shape_vec((height as usize, width as usize, 3), raw)
                .map_err(|e| PyValueError::new_err(format!("Failed to reshape array: {}", e)))?;
            Ok(Some(array.into_pyarray(py)))
        }
        Ok(None) => Ok(None),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Check if a format is supported by the native decoder.
#[pyfunction]
fn is_format_supported(data: &[u8]) -> bool {
    image::guess_format(data).is_ok()
}

#[pymodule]
fn kestrel_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_image, m)?)?;
    m.add_function(wrap_pyfunction!(is_format_supported, m)?)?;
    Ok(())
}
