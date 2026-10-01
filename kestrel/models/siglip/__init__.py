"""The Moondream-trained SigLIP SO400M/14 image encoder."""
from kestrel.models.registry import register_lazy

register_lazy(["siglip-so400m-378"], __name__ + ".registration")
