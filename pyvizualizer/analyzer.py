# pyvizualizer/analyzer.py

import logging
from .utils import get_python_files
from .parser import CodeParser
from .graph import CodeGraph
from .exceptions import PyVizualizerError

logger = logging.getLogger('pyvizualizer')

class Analyzer:
    """Main class responsible for analyzing a Python project."""

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.parser = CodeParser()
        self.graph = CodeGraph()

    def analyze(self):
        """Performs the analysis of the project."""
        logger.info(f"Starting analysis of project at: {self.project_path}")
        python_files = get_python_files(self.project_path)
        logger.debug(f"Python files found: {python_files}")
        self.parser.parse_files(python_files)
        self.graph.build_graph(self.parser.definitions)
        logger.info("Analysis complete")
        return self.graph
