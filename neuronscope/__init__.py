"""NeuronScope: trace which neurons and attention heads drove a language model's output."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("neuronscope-cli")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
