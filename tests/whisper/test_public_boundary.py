from __future__ import annotations

import ast
from pathlib import Path


def test_public_model_does_not_import_a_backend_implementation() -> None:
    source_root = Path(__file__).parents[2] / "kestrel" / "models" / "whisper"
    imported: set[str] = set()
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

    backend_implementation_imports = {
        name for name in imported if name == "mkl" or name.startswith("mkl.")
    }
    assert not backend_implementation_imports
    assert not {
        name
        for name in imported
        if name == "kestrel_whisper" or name.startswith("kestrel_whisper.")
    }
    assert not (source_root / "native_backend.py").exists()
    assert not (source_root / "generated_session.py").exists()
    assert not (source_root / "decode_trace.py").exists()
