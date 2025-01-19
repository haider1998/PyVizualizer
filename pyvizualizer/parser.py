# pyvizualizer/parser.py

import ast
from collections import defaultdict
from typing import Dict, Set
import logging

from pyvizualizer.exceptions import FileParsingError

logger = logging.getLogger('pyvizualizer')

class CodeParser:
    """Parses Python code to extract class and function definitions."""

    def __init__(self):
        self.definitions = defaultdict(dict)  # module: {definitions}

    def parse_files(self, file_paths):
        """Parses multiple Python files."""
        for file_path in file_paths:
            print(f"Parsing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    source = file.read()
                tree = ast.parse(source, filename=file_path)
                self._parse_tree(tree, file_path)
            except (SyntaxError, FileNotFoundError) as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                raise FileParsingError(f"Failed to parse {file_path}") from e

    def _parse_tree(self, tree, file_path):
        """Parses an AST tree to extract definitions."""
        visitor = _DefinitionVisitor(file_path)
        visitor.visit(tree)
        self.definitions[file_path] = visitor.definitions


class _DefinitionVisitor(ast.NodeVisitor):
    """AST visitor that records definitions and call relationships."""

    def __init__(self, module_name):
        self.module_name = module_name
        self.definitions = []
        self.current_class = None
        self.current_function = None

    def visit_ClassDef(self, node):
        self.current_class = node.name
        print(f"Found class: {self.current_class}")
        class_info = {
            'type': 'class',
            'name': self.current_class,
            'methods': [],
        }
        self.definitions.append(class_info)
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        function_name = node.name
        if self.current_class:
            print(f"Found method: {function_name} in class {self.current_class}")
            # Find the class info in definitions
            for item in self.definitions:
                if item['type'] == 'class' and item['name'] == self.current_class:
                    method_info = {
                        'name': function_name,
                        'calls': self._extract_calls(node)
                    }
                    item['methods'].append(method_info)
                    break
        else:
            print(f"Found function: {function_name}")
            function_info = {
                'type': 'function',
                'name': function_name,
                'calls': self._extract_calls(node)
            }
            self.definitions.append(function_info)
        self.generic_visit(node)

    def _extract_calls(self, node):
        """Extracts function calls within a function/method body."""
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
        return list(calls)