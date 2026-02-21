#!/usr/bin/env python3
"""Test client for Kestrel Triton backend.

Exercises all four skills (query, caption, detect, point) in both streaming
and non-streaming modes against the unified ``moondream`` Triton model.

Usage:
    pip install tritonclient[grpc] numpy
    python triton/client.py [--url localhost:8001] [--image path/to/image.jpg]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tritonclient.grpc as grpcclient

MODEL_NAME = "moondream"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_input(name: str, data: bytes) -> grpcclient.InferInput:
    inp = grpcclient.InferInput(name, [1], "BYTES")
    inp.set_data_from_numpy(np.array([data], dtype=np.object_))
    return inp


def _string_input(name: str, value: str) -> grpcclient.InferInput:
    inp = grpcclient.InferInput(name, [1], "BYTES")
    inp.set_data_from_numpy(np.array([value], dtype=np.object_))
    return inp


def _bool_input(name: str, value: bool) -> grpcclient.InferInput:
    inp = grpcclient.InferInput(name, [1], "BOOL")
    inp.set_data_from_numpy(np.array([value], dtype=np.bool_))
    return inp


def _int_input(name: str, value: int) -> grpcclient.InferInput:
    inp = grpcclient.InferInput(name, [1], "INT32")
    inp.set_data_from_numpy(np.array([value], dtype=np.int32))
    return inp


def _parse_output(result: Any) -> Dict[str, Any]:
    raw = result.as_numpy("TEXT_OUTPUT")[0]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return json.loads(text)


def _print_result(label: str, data: Dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2))


def _print_pass(label: str) -> None:
    print(f"  PASS  {label}")


def _print_fail(label: str, reason: str) -> None:
    print(f"  FAIL  {label}: {reason}")


def _infer_decoupled(
    client_url: str,
    inputs: List[grpcclient.InferInput],
    timeout: float = 120,
) -> List[Dict[str, Any]]:
    """Send a request via streaming API (required for decoupled models).

    Returns all response chunks collected.
    """
    chunks: List[Dict[str, Any]] = []
    errors: List[str] = []
    done = threading.Event()

    def callback(result, error):
        if error:
            errors.append(str(error))
            done.set()
            return
        data = _parse_output(result)
        chunks.append(data)
        if data.get("completed", True):
            # For non-streaming skills there's only one response (always "final")
            done.set()

    client = grpcclient.InferenceServerClient(client_url)
    client.start_stream(callback=callback)
    client.async_stream_infer(MODEL_NAME, inputs)
    done.wait(timeout=timeout)
    client.stop_stream()

    if errors:
        raise RuntimeError(f"Server errors: {errors}")
    return chunks


# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------

def test_query(client_url: str, image: bytes) -> bool:
    label = "query (non-streaming)"
    try:
        inputs = [
            _string_input("SKILL", "query"),
            _string_input("QUESTION", "Describe what you see in this image in one sentence."),
            _bytes_input("IMAGE", image),
        ]
        chunks = _infer_decoupled(client_url, inputs)
        assert len(chunks) == 1, f"expected 1 response, got {len(chunks)}"
        data = chunks[0]
        _print_result(label, data)
        assert "answer" in data, "missing 'answer' field"
        assert data.get("metrics"), "missing 'metrics'"
        _print_pass(label)
        return True
    except Exception as exc:
        _print_fail(label, str(exc))
        return False


def test_query_streaming(client_url: str, image: bytes) -> bool:
    label = "query (streaming)"
    try:
        inputs = [
            _string_input("SKILL", "query"),
            _string_input("QUESTION", "What colors do you see in this image?"),
            _bytes_input("IMAGE", image),
            _bool_input("STREAM", True),
        ]
        chunks: List[Dict[str, Any]] = []
        errors: List[str] = []
        done = threading.Event()

        def callback(result, error):
            if error:
                errors.append(str(error))
                done.set()
                return
            data = _parse_output(result)
            chunks.append(data)
            if data.get("completed"):
                done.set()

        client = grpcclient.InferenceServerClient(client_url)
        client.start_stream(callback=callback)
        client.async_stream_infer(MODEL_NAME, inputs)
        done.wait(timeout=120)
        client.stop_stream()

        if errors:
            _print_fail(label, f"errors: {errors}")
            return False

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Received {len(chunks)} chunks")
        final = chunks[-1] if chunks else {}
        print(f"  Final answer: {final.get('answer', '???')[:100]}")
        print(f"  Metrics: {json.dumps(final.get('metrics', {}), indent=2)}")

        assert len(chunks) >= 1, "expected at least 1 chunk"
        assert final.get("completed"), "last chunk should have completed=True"
        assert "answer" in final, "final chunk missing 'answer'"
        _print_pass(label)
        return True
    except Exception as exc:
        _print_fail(label, str(exc))
        return False


def test_caption(client_url: str, image: bytes) -> bool:
    label = "caption (non-streaming)"
    try:
        inputs = [
            _string_input("SKILL", "caption"),
            _bytes_input("IMAGE", image),
            _string_input("LENGTH", "short"),
        ]
        chunks = _infer_decoupled(client_url, inputs)
        assert len(chunks) == 1, f"expected 1 response, got {len(chunks)}"
        data = chunks[0]
        _print_result(label, data)
        assert "caption" in data, "missing 'caption' field"
        _print_pass(label)
        return True
    except Exception as exc:
        _print_fail(label, str(exc))
        return False


def test_caption_streaming(client_url: str, image: bytes) -> bool:
    label = "caption (streaming)"
    try:
        inputs = [
            _string_input("SKILL", "caption"),
            _bytes_input("IMAGE", image),
            _string_input("LENGTH", "normal"),
            _bool_input("STREAM", True),
        ]
        chunks: List[Dict[str, Any]] = []
        errors: List[str] = []
        done = threading.Event()

        def callback(result, error):
            if error:
                errors.append(str(error))
                done.set()
                return
            data = _parse_output(result)
            chunks.append(data)
            if data.get("completed"):
                done.set()

        client = grpcclient.InferenceServerClient(client_url)
        client.start_stream(callback=callback)
        client.async_stream_infer(MODEL_NAME, inputs)
        done.wait(timeout=120)
        client.stop_stream()

        if errors:
            _print_fail(label, f"errors: {errors}")
            return False

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Received {len(chunks)} chunks")
        final = chunks[-1] if chunks else {}
        print(f"  Final caption: {final.get('caption', '???')[:100]}")
        print(f"  Metrics: {json.dumps(final.get('metrics', {}), indent=2)}")

        assert len(chunks) >= 1, "expected at least 1 chunk"
        assert final.get("completed"), "last chunk should have completed=True"
        assert "caption" in final, "final chunk missing 'caption'"
        _print_pass(label)
        return True
    except Exception as exc:
        _print_fail(label, str(exc))
        return False


def test_detect(client_url: str, image: bytes) -> bool:
    label = "detect"
    try:
        inputs = [
            _string_input("SKILL", "detect"),
            _string_input("OBJECT", "face"),
            _bytes_input("IMAGE", image),
        ]
        chunks = _infer_decoupled(client_url, inputs)
        assert len(chunks) == 1, f"expected 1 response, got {len(chunks)}"
        data = chunks[0]
        _print_result(label, data)
        assert "objects" in data, "missing 'objects' field"
        _print_pass(label)
        return True
    except Exception as exc:
        _print_fail(label, str(exc))
        return False


def test_point(client_url: str, image: bytes) -> bool:
    label = "point"
    try:
        inputs = [
            _string_input("SKILL", "point"),
            _string_input("OBJECT", "face"),
            _bytes_input("IMAGE", image),
        ]
        chunks = _infer_decoupled(client_url, inputs)
        assert len(chunks) == 1, f"expected 1 response, got {len(chunks)}"
        data = chunks[0]
        _print_result(label, data)
        assert "points" in data, "missing 'points' field"
        _print_pass(label)
        return True
    except Exception as exc:
        _print_fail(label, str(exc))
        return False


def test_error_missing_required(client_url: str) -> bool:
    """Send a query request without the required QUESTION input."""
    label = "error (missing required input)"
    try:
        inputs = [
            _string_input("SKILL", "query"),
            _bytes_input("IMAGE", b"fake"),
        ]
        chunks: List[Dict[str, Any]] = []
        errors: List[str] = []
        done = threading.Event()

        def callback(result, error):
            if error:
                errors.append(str(error))
            else:
                data = _parse_output(result)
                chunks.append(data)
            done.set()

        client = grpcclient.InferenceServerClient(client_url)
        client.start_stream(callback=callback)
        client.async_stream_infer(MODEL_NAME, inputs)
        done.wait(timeout=30)
        client.stop_stream()

        if errors:
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            print(f"  Got expected error: {errors[0]}")
            _print_pass(label)
            return True
        else:
            _print_fail(label, "expected error but got a response")
            return False
    except Exception as exc:
        _print_fail(label, str(exc))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Kestrel Triton test client")
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--image", default=None, help="Path to test image (jpg/png/webp)")
    args = parser.parse_args()

    # Load test image
    if args.image:
        image_path = Path(args.image)
    else:
        # Try a well-known test image from the repo
        candidates = [
            Path(__file__).resolve().parent.parent / "assets" / "test.jpg",
            Path(__file__).resolve().parent.parent / "assets" / "kestrel-overview.png",
        ]
        image_path = None
        for c in candidates:
            if c.exists():
                image_path = c
                break
        if image_path is None:
            # Download a small test image
            import urllib.request
            image_path = Path("/tmp/triton_test_image.jpg")
            if not image_path.exists():
                print("Downloading test image...")
                urllib.request.urlretrieve(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/481px-Cat_November_2010-1a.jpg",
                    str(image_path),
                )

    image_bytes = image_path.read_bytes()
    print(f"Using test image: {image_path} ({len(image_bytes)} bytes)")

    client = grpcclient.InferenceServerClient(args.url)

    # Check server is ready
    if not client.is_server_live():
        print("ERROR: Triton server is not live")
        sys.exit(1)
    print("Triton server is live\n")

    results = []
    t0 = time.time()

    # Non-streaming tests
    results.append(("query", test_query(args.url, image_bytes)))
    results.append(("caption", test_caption(args.url, image_bytes)))
    results.append(("detect", test_detect(args.url, image_bytes)))
    results.append(("point", test_point(args.url, image_bytes)))

    # Streaming tests
    results.append(("query_stream", test_query_streaming(args.url, image_bytes)))
    results.append(("caption_stream", test_caption_streaming(args.url, image_bytes)))

    # Error test
    results.append(("error", test_error_missing_required(args.url)))

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({elapsed:.1f}s)")
    print(f"{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
