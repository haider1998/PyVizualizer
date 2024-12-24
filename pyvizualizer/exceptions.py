# pyvizualizer/exceptions.py

class PyVizualizerError(Exception):
    """Base exception class for PyVizualizer errors."""
    pass


class FileParsingError(PyVizualizerError):
    """Exception raised when a file cannot be parsed."""
    pass


class GraphConstructionError(PyVizualizerError):
    """Exception raised during graph construction."""
    pass


class MermaidGenerationError(PyVizualizerError):
    """Exception raised during Mermaid code generation."""
    pass
