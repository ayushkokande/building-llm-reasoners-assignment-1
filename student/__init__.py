import importlib.metadata

try:
    __version__ = importlib.metadata.version("student")
except importlib.metadata.PackageNotFoundError:
    # Not installed as a distribution (e.g. run as a plain source folder on
    # HuggingFace Spaces). Version metadata is not needed at runtime.
    __version__ = "0+local"
