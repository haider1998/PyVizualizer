# PyVizualizer

PyVizualizer is a tool that analyzes Python projects and generates visual diagrams (in Mermaid format) representing the workflow and method/class interactions within the project, excluding external packages.

## Features

- Parses your Python project's code using the `ast` module.
- Constructs a graph representing classes, methods, and their interactions.
- Generates Mermaid code that can be converted into visual diagrams.
- Command-line interface for easy usage.
- Future scope to generate SVG images and integrate into documentation.

## Installation

```bash
pip install pyvizualizer
