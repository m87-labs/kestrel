"""HTTP server interface for the Kestrel inference engine."""

from .http import create_app

__all__ = ["create_app"]
