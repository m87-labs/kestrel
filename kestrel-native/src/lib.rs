use fast_image_resize::{images::Image, FilterType, ResizeAlg, ResizeOptions, Resizer};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3, PyUntypedArrayMethods};
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

/// Wrapper for numpy array data that can be sent across threads.
struct SendableArray {
    ptr: *const u8,
    height: usize,
    width: usize,
    channels: usize,
}

unsafe impl Send for SendableArray {}
unsafe impl Sync for SendableArray {}

impl SendableArray {
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.height * self.width * self.channels) }
    }
}

/// Resize an RGB image using Lanczos3 interpolation.
///
/// Args:
///     image: Input image as numpy array (H, W, 3) in uint8 RGB format.
///     target_height: Target height in pixels.
///     target_width: Target width in pixels.
///
/// Returns:
///     Resized image as numpy array (target_height, target_width, 3).
#[pyfunction]
fn resize_lanczos3<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, u8>,
    target_height: u32,
    target_width: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let shape = image.shape();
    let (src_height, src_width, channels) = (shape[0], shape[1], shape[2]);

    if channels != 3 {
        return Err(PyValueError::new_err(format!(
            "Expected 3 channels (RGB), got {}",
            channels
        )));
    }

    let slice = image.as_slice().map_err(|e| {
        PyValueError::new_err(format!("Array must be contiguous: {}", e))
    })?;

    let sendable = SendableArray {
        ptr: slice.as_ptr(),
        height: src_height,
        width: src_width,
        channels,
    };

    let result = py.allow_threads(move || -> Result<Vec<u8>, String> {
        let src_data = sendable.as_slice();

        if sendable.width == 0 || sendable.height == 0 {
            return Err("Source dimensions must be non-zero".to_string());
        }
        if target_width == 0 || target_height == 0 {
            return Err("Target dimensions must be non-zero".to_string());
        }

        let src_image = Image::from_slice_u8(
            sendable.width as u32,
            sendable.height as u32,
            // SAFETY: We know the slice is valid for the dimensions
            unsafe { std::slice::from_raw_parts_mut(src_data.as_ptr() as *mut u8, src_data.len()) },
            fast_image_resize::PixelType::U8x3,
        ).map_err(|e| format!("Failed to create source image: {}", e))?;

        let mut dst_image = Image::new(target_width, target_height, fast_image_resize::PixelType::U8x3);

        let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
        let mut resizer = Resizer::new();
        resizer
            .resize(&src_image, &mut dst_image, &options)
            .map_err(|e| format!("Failed to resize: {}", e))?;

        Ok(dst_image.into_vec())
    });

    match result {
        Ok(data) => {
            let array = ndarray::Array3::from_shape_vec(
                (target_height as usize, target_width as usize, 3),
                data,
            )
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape array: {}", e)))?;
            Ok(array.into_pyarray(py))
        }
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pymodule]
fn kestrel_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_image, m)?)?;
    m.add_function(wrap_pyfunction!(is_format_supported, m)?)?;
    m.add_function(wrap_pyfunction!(resize_lanczos3, m)?)?;
    Ok(())
}
