"""
Python Method Visualization Tool

A tool to visualize the relationships between methods and functions in Python projects
using Mermaid diagrams, focusing solely on project-defined methods.
"""

import ast
import os
import sys
import argparse
import networkx as nx
from pathlib import Path
import logging
import re
import concurrent.futures
from functools import lru_cache
from typing import Dict, List, Set, Optional, Tuple, Any, Union, NamedTuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("method-visualizer")


class ImportInfo(NamedTuple):
    """Data structure to store information about imported names."""
    module: str
    name: str
    alias: str
    is_star: bool = False


class ImportCollector(ast.NodeVisitor):
    """AST visitor to collect all imports before function analysis."""

    def __init__(self, current_module: str, project_root: str):
        self.current_module = current_module
        self.project_root = project_root
        self.import_map = {}  # Maps imported names to their module
        self.import_from_map = {}  # Maps module names to the set of names imported from them
        self.direct_imports = set()  # Set of directly imported modules
        self.all_modules = set()  # All modules encountered
        self.star_imports = set()  # Modules from which * was imported
        # Store import information with more details
        self.imports = []  # List of ImportInfo objects
        # Cache to avoid resolving the same modules repeatedly
        self.resolved_modules = {}

    def visit_Import(self, node):
        """Process import statements."""
        for name in node.names:
            module_name = name.name
            alias = name.asname or module_name

            # Record the import
            self.import_map[alias] = module_name
            self.direct_imports.add(module_name)
            self.all_modules.add(module_name)

            # Add to detailed imports list
            self.imports.append(ImportInfo(module_name, module_name, alias))

            # Also record the module components for qualified name resolution
            parts = module_name.split('.')
            for i in range(1, len(parts) + 1):
                partial_module = '.'.join(parts[:i])
                self.all_modules.add(partial_module)

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Process from-import statements."""
        if node.module is not None or node.level > 0:
            # Handle relative imports by resolving the relative path
            if node.level > 0:
                module_name = self._resolve_relative_import(node.module, node.level)
            else:
                module_name = node.module

            self.all_modules.add(module_name)

            if module_name not in self.import_from_map:
                self.import_from_map[module_name] = set()

            for name in node.names:
                if name.name == '*':
                    # Import all names - we'll need to resolve this later
                    self.import_from_map[module_name].add('*')
                    self.star_imports.add(module_name)
                    self.imports.append(ImportInfo(module_name, '*', '*', True))
                else:
                    imported_name = name.name
                    alias = name.asname or imported_name

                    # Record this specific import
                    self.import_map[alias] = f"{module_name}.{imported_name}"
                    self.import_from_map[module_name].add(imported_name)
                    self.imports.append(ImportInfo(module_name, imported_name, alias))

        self.generic_visit(node)

    def _resolve_relative_import(self, module_name: Optional[str], level: int) -> str:
        """Resolve a relative import to an absolute module path."""
        if level == 0:
            return module_name or ""

        # Get the current package parts
        parts = self.current_module.split('.')

        # For relative imports, go up 'level' packages
        if len(parts) < level:
            logger.warning(f"Invalid relative import in {self.current_module}: level {level} too high")
            # Return best effort
            package = ""
        else:
            package = '.'.join(parts[:-level])

        # Add the specified module if any
        if module_name:
            if package:
                return f"{package}.{module_name}"
            return module_name
        return package


class ModuleAnalyzer:
    """Manages analysis of an entire module including imports and function definitions."""

    def __init__(self, module_name: str, file_path: str, tree: ast.AST, project_root: str):
        self.module_name = module_name
        self.file_path = file_path
        self.tree = tree
        self.project_root = project_root
        self.imports = ImportCollector(module_name, project_root)
        self.imports.visit(tree)

        # Maps class names to their definitions with inheritance info
        self.classes = {}
        # Maps function/method names to their definitions
        self.functions = {}
        # Maps class variables and function variables to what they reference
        self.variable_map = {}
        # Track all method calls
        self.calls = []
        # Track decorator usage
        self.decorators = {}
        # Track type annotations
        self.type_annotations = {}

        # Process class and function definitions
        self._collect_definitions()

    def _collect_definitions(self):
        """Collect all class and function definitions from the module."""
        class_stack = []  # Track nested classes

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                # Build full class name with proper nesting
                parent_prefix = f"{class_stack[-1]}." if class_stack else self.module_name + "."
                full_class_name = f"{parent_prefix}{node.name}"
                class_stack.append(full_class_name)

                # Process inheritance
                base_classes = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_classes.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        # Handle module.Class inheritance
                        parts = self._extract_attribute_chain(base)
                        if parts:
                            base_classes.append('.'.join(parts))

                self.classes[full_class_name] = {
                    'name': node.name,
                    'module': self.module_name,
                    'bases': base_classes,
                    'methods': {},
                    'node': node,
                    'decorators': [self._process_decorator(d) for d in node.decorator_list]
                }

                # Collect methods in the class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                        method_name = f"{full_class_name}.{item.name}"

                        # Check if this is a property
                        is_property = any(
                            d.id == 'property' if isinstance(d, ast.Name) else False
                            for d in item.decorator_list
                        )

                        self.classes[full_class_name]['methods'][item.name] = {
                            'name': item.name,
                            'full_name': method_name,
                            'node': item,
                            'is_async': isinstance(item, ast.AsyncFunctionDef),
                            'is_property': is_property,
                            'decorators': [self._process_decorator(d) for d in item.decorator_list],
                            'return_annotation': self._process_annotation(item.returns) if item.returns else None
                        }

                        # Extract argument types if available
                        arg_types = {}
                        for arg in item.args.args:
                            if arg.annotation:
                                arg_types[arg.arg] = self._process_annotation(arg.annotation)

                        # Also add to functions map for consistency
                        self.functions[method_name] = {
                            'name': item.name,
                            'module': self.module_name,
                            'class': full_class_name,
                            'full_name': method_name,
                            'lineno': item.lineno,
                            'node': item,
                            'is_async': isinstance(item, ast.AsyncFunctionDef),
                            'is_property': is_property,
                            'decorators': [self._process_decorator(d) for d in item.decorator_list],
                            'return_annotation': self._process_annotation(item.returns) if item.returns else None,
                            'arg_types': arg_types
                        }

                # After processing the class, remove it from the stack
                class_stack.pop()

            elif (isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)) and not class_stack:
                # Skip methods, we handle them above
                func_name = f"{self.module_name}.{node.name}"

                # Extract argument types if available
                arg_types = {}
                for arg in node.args.args:
                    if arg.annotation:
                        arg_types[arg.arg] = self._process_annotation(arg.annotation)

                self.functions[func_name] = {
                    'name': node.name,
                    'module': self.module_name,
                    'class': None,
                    'full_name': func_name,
                    'lineno': node.lineno,
                    'node': node,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'is_property': False,
                    'decorators': [self._process_decorator(d) for d in node.decorator_list],
                    'return_annotation': self._process_annotation(node.returns) if node.returns else None,
                    'arg_types': arg_types
                }

    def _extract_attribute_chain(self, node: ast.AST) -> List[str]:
        """Extract a chain of attribute access like module.submodule.Class."""
        parts = []

        # Handle the base case of a Name node
        if isinstance(node, ast.Name):
            return [node.id]

        # Handle Attribute nodes recursively
        elif isinstance(node, ast.Attribute):
            # Get the value parts recursively
            value_parts = self._extract_attribute_chain(node.value)
            # Add the attribute
            return value_parts + [node.attr]

        return parts

    def _process_decorator(self, node: ast.AST) -> Dict[str, Any]:
        """Process a decorator node and extract information."""
        if isinstance(node, ast.Name):
            # Simple decorator: @decorator_name
            return {'type': 'name', 'name': node.id}
        elif isinstance(node, ast.Call):
            # Decorator with arguments: @decorator(args)
            if isinstance(node.func, ast.Name):
                return {'type': 'call', 'name': node.func.id, 'args': self._extract_call_args(node)}
            elif isinstance(node.func, ast.Attribute):
                # Qualified decorator: @module.decorator(args)
                parts = self._extract_attribute_chain(node.func)
                return {'type': 'call', 'name': '.'.join(parts), 'args': self._extract_call_args(node)}
        elif isinstance(node, ast.Attribute):
            # Qualified decorator: @module.decorator
            parts = self._extract_attribute_chain(node)
            return {'type': 'name', 'name': '.'.join(parts)}

        # Default for unknown decorator types
        return {'type': 'unknown'}

    def _extract_call_args(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract arguments from a function call."""
        args = {}

        # Process positional arguments
        if call_node.args:
            args['positional'] = [
                self._extract_arg_value(arg) for arg in call_node.args
            ]

        # Process keyword arguments
        if call_node.keywords:
            args['keywords'] = {
                kw.arg: self._extract_arg_value(kw.value) for kw in call_node.keywords
            }

        return args

    def _extract_arg_value(self, node: ast.AST) -> Any:
        """Extract a simple value from an AST node if possible."""
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value  # True, False, None
        elif isinstance(node, ast.Name):
            return f"variable:{node.id}"
        elif isinstance(node, ast.Attribute):
            parts = self._extract_attribute_chain(node)
            return '.'.join(parts)
        # For other types, return a placeholder
        return "complex_value"

    def _process_annotation(self, node: ast.AST) -> Dict[str, Any]:
        """Process a type annotation node."""
        if isinstance(node, ast.Name):
            # Simple annotation: int, str, etc.
            return {'type': 'name', 'name': node.id}
        elif isinstance(node, ast.Attribute):
            # Qualified annotation: module.Class
            parts = self._extract_attribute_chain(node)
            return {'type': 'name', 'name': '.'.join(parts)}
        elif isinstance(node, ast.Subscript):
            # Generic type: List[int], Dict[str, int], etc.
            if isinstance(node.value, ast.Name):
                container = node.value.id
            elif isinstance(node.value, ast.Attribute):
                parts = self._extract_attribute_chain(node.value)
                container = '.'.join(parts)
            else:
                container = "unknown"

            # Extract the parameters if possible
            params = []
            if isinstance(node.slice, ast.Index):  # Python 3.8 and below
                slice_value = node.slice.value
            else:  # Python 3.9+
                slice_value = node.slice

            if isinstance(slice_value, ast.Tuple):
                for elt in slice_value.elts:
                    params.append(self._process_annotation(elt))
            else:
                params.append(self._process_annotation(slice_value))

            return {'type': 'subscript', 'container': container, 'params': params}

        # For other types, return a placeholder
        return {'type': 'unknown'}


class FunctionCallVisitor(ast.NodeVisitor):
    """AST visitor to extract function calls."""

    def __init__(self, module_name: str, file_path: str, module_analyzer: ModuleAnalyzer,
                 all_modules: Dict[str, ModuleAnalyzer], all_module_names: Set[str]):
        self.current_function = None
        self.current_class = None
        self.class_stack = []  # Track nested classes
        self.function_stack = []  # Track nested functions
        self.module_name = module_name
        self.file_path = file_path
        self.module_analyzer = module_analyzer
        self.all_modules = all_modules
        self.all_module_names = all_module_names
        self.calls = []
        self.class_instances = {}  # Maps variable names to class types
        self.current_class_vars = {}  # Maps 'self.var' to their types in current class
        self.context_managers = {}  # Track variables created in context managers

        # Cache to avoid repeated lookups
        self._method_cache = {}
        self._class_cache = {}

    def visit_ClassDef(self, node):
        """Visit a class definition."""
        previous_class = self.current_class
        # Handle nested classes
        if self.class_stack:
            parent_class = self.class_stack[-1]
            self.current_class = f"{parent_class}.{node.name}"
        else:
            self.current_class = f"{self.module_name}.{node.name}"

        self.class_stack.append(self.current_class)
        previous_vars = self.current_class_vars.copy()
        self.current_class_vars = {}  # Reset for this class

        # Visit all children to find methods and assignments
        self.generic_visit(node)

        # Restore context
        self.class_stack.pop()
        self.current_class = previous_class
        self.current_class_vars = previous_vars

    def visit_FunctionDef(self, node):
        """Visit a function definition."""
        self._visit_function_common(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit an async function definition."""
        self._visit_function_common(node)

    def _visit_function_common(self, node):
        """Common handling for function and async function definitions."""
        parent_func = self.current_function

        # Build the full function name based on context
        if self.class_stack:
            self.current_function = f"{self.class_stack[-1]}.{node.name}"
        else:
            self.current_function = f"{self.module_name}.{node.name}"

        self.function_stack.append(self.current_function)

        # Process decorators before visiting the function body
        for decorator in node.decorator_list:
            self.visit(decorator)

        # Visit all children to find calls
        self.generic_visit(node)

        # Restore parent function context
        self.function_stack.pop()
        self.current_function = parent_func if self.function_stack else None

    def visit_Assign(self, node):
        """Visit an assignment statement to track variables."""
        # Only process if we have a value that is potentially a class instance
        if isinstance(node.value, ast.Call):
            # Get the class name being instantiated
            class_name = self._extract_call_target(node.value)

            # Record the variable assignment
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    if class_name:
                        self.class_instances[var_name] = class_name
                elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    # Handle self.var = something()
                    if target.value.id == 'self' and self.current_class:
                        var_name = f"self.{target.attr}"
                        if class_name:
                            self.current_class_vars[var_name] = class_name

        # Handle tuple unpacking and other assignment types
        elif isinstance(node.value, ast.Tuple):
            # Check each item in the tuple for potential class instances
            for i, elt in enumerate(node.value.elts):
                if isinstance(elt, ast.Call):
                    class_name = self._extract_call_target(elt)
                    if class_name and i < len(node.targets):
                        target = node.targets[i]
                        if isinstance(target, ast.Name):
                            self.class_instances[target.id] = class_name

        # Handle walrus operator in Python 3.8+
        elif isinstance(node, ast.NamedExpr) and isinstance(node.value, ast.Call):
            class_name = self._extract_call_target(node.value)
            if class_name and isinstance(node.target, ast.Name):
                self.class_instances[node.target.id] = class_name

        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """Visit an annotated assignment (variable with type hint)."""
        # Record type annotation
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            if node.annotation:
                # This is a type hint that can help with resolution
                annotation = self._process_annotation(node.annotation)
                if annotation:
                    self.class_instances[var_name] = annotation

        # If there's a value being assigned, process it as well
        if node.value and isinstance(node.value, ast.Call):
            class_name = self._extract_call_target(node.value)
            if class_name and isinstance(node.target, ast.Name):
                self.class_instances[node.target.id] = class_name

        self.generic_visit(node)

    def visit_With(self, node):
        """Visit a with statement to track context managers."""
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                # The context manager expression is a call
                class_name = self._extract_call_target(item.context_expr)

                # If there's an optional_vars (the 'as' part)
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    var_name = item.optional_vars.id
                    if class_name:
                        self.context_managers[var_name] = class_name
                        # Also add to class instances for method resolution
                        self.class_instances[var_name] = class_name

        self.generic_visit(node)

    def visit_Call(self, node):
        """Visit a function call."""
        if not self.current_function:
            # Skip calls outside of function definitions
            self.generic_visit(node)
            return

        # Extract call target and potential module/class context
        target_function, module_path = self._resolve_call(node)

        if target_function:
            self.record_call(self.current_function, target_function, node.lineno)

        # Don't forget to visit function arguments which might contain calls
        for arg in node.args:
            self.visit(arg)

        for keyword in node.keywords:
            self.visit(keyword.value)

    def _resolve_call(self, node: ast.Call) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a function call to its fully qualified name if possible."""
        if isinstance(node.func, ast.Name):
            # Direct function call: function_name()
            func_name = node.func.id

            # Check if it's an imported name
            if func_name in self.module_analyzer.imports.import_map:
                imported_path = self.module_analyzer.imports.import_map[func_name]

                # Check if this points to a function in a module
                if '.' in imported_path:
                    module_path, func_name = imported_path.rsplit('.', 1)
                    # Look for this function in all modules
                    for m_name, analyzer in self.all_modules.items():
                        if m_name == module_path or m_name.endswith('.' + module_path):
                            target = f"{m_name}.{func_name}"
                            if target in analyzer.functions:
                                return target, module_path

                    # If we didn't find a direct match, this might be a module.function reference
                    return imported_path, None
                else:
                    # It's a module import with alias, but we can't resolve the function
                    return None, imported_path

            # Check if it's a local function in current module
            local_target = f"{self.module_name}.{func_name}"
            if local_target in self.module_analyzer.functions:
                return local_target, None

            # Check if it might be in any of the star-imported modules
            for star_module in self.module_analyzer.imports.star_imports:
                for m_name, analyzer in self.all_modules.items():
                    if m_name == star_module or m_name.endswith('.' + star_module):
                        candidate = f"{m_name}.{func_name}"
                        if candidate in analyzer.functions:
                            return candidate, star_module

            # If we can't resolve it, just return the name for later resolution
            return func_name, None

        elif isinstance(node.func, ast.Attribute):
            # Method or attribute call: object.method()
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr

                # Handle self.method() calls
                if obj_name == 'self' and self.current_class:
                    # Self method call within a class
                    method_target = f"{self.current_class}.{method_name}"

                    # Check if the method exists in this class
                    if self.current_class in self.module_analyzer.classes:
                        if method_name in self.module_analyzer.classes[self.current_class]['methods']:
                            return method_target, None

                    # Method might be inherited, try to find it in base classes
                    class_info = self.module_analyzer.classes.get(self.current_class)
                    if class_info and class_info['bases']:
                        for base_class in class_info['bases']:
                            # Resolve the base class name to a full qualified name if needed
                            full_base_name = self._resolve_class_name(base_class)
                            if full_base_name:
                                base_target = f"{full_base_name}.{method_name}"
                                # Check if this method exists in any of the analyzers
                                for analyzer in self.all_modules.values():
                                    if base_target in analyzer.functions:
                                        return base_target, None

                    # Method might be inherited, but we'll still record the call
                    return method_target, None

                # Handle calls on class instances
                if obj_name in self.class_instances:
                    class_name = self.class_instances[obj_name]

                    # Cache key for performance
                    cache_key = f"{class_name}::{method_name}"
                    if cache_key in self._method_cache:
                        return self._method_cache[cache_key]

                    # Try to find the class definition
                    for m_name, analyzer in self.all_modules.items():
                        for c_name, c_info in analyzer.classes.items():
                            c_short_name = c_info['name']
                            if class_name.endswith('.' + c_short_name) or class_name == c_short_name:
                                # Found the class, now check for the method
                                if method_name in c_info['methods']:
                                    result = (f"{c_name}.{method_name}", None)
                                    self._method_cache[cache_key] = result
                                    return result

                                # Check base classes for inherited methods
                                for base_class in c_info.get('bases', []):
                                    full_base_name = self._resolve_class_name(base_class)
                                    if full_base_name:
                                        for analyzer2 in self.all_modules.values():
                                            if full_base_name in analyzer2.classes:
                                                base_info = analyzer2.classes[full_base_name]
                                                if method_name in base_info['methods']:
                                                    result = (f"{full_base_name}.{method_name}", None)
                                                    self._method_cache[cache_key] = result
                                                    return result

                    # If can't find exact method, return with class name for later resolution
                    result = (f"{class_name}.{method_name}", None)
                    self._method_cache[cache_key] = result
                    return result

                # Handle module method calls
                if obj_name in self.module_analyzer.imports.import_map:
                    module_path = self.module_analyzer.imports.import_map[obj_name]

                    # Check if this might be a module.function call
                    for m_name, analyzer in self.all_modules.items():
                        if m_name == module_path or m_name.endswith('.' + module_path):
                            target = f"{m_name}.{method_name}"
                            if target in analyzer.functions:
                                return target, module_path

                    # If not found, return for later resolution
                    return f"{module_path}.{method_name}", module_path

                # Check if this is a context manager variable
                if obj_name in self.context_managers:
                    class_name = self.context_managers[obj_name]
                    return f"{class_name}.{method_name}", None

            # Handle nested attribute access like module.submodule.function()
            elif isinstance(node.func.value, ast.Attribute):
                # Extract the full attribute chain
                parts = self._extract_attribute_chain(node.func)
                if len(parts) >= 2:
                    obj_path = '.'.join(parts[:-1])
                    method_name = parts[-1]

                    # Check if the parts correspond to a known module or class
                    for m_name, analyzer in self.all_modules.items():
                        # Check if the object path matches or ends with a module name
                        if m_name == obj_path or m_name.endswith('.' + obj_path):
                            target = f"{m_name}.{method_name}"
                            if target in analyzer.functions:
                                return target, obj_path

                    # Return best effort result for later resolution
                    return f"{obj_path}.{method_name}", obj_path

        return None, None

    def _resolve_class_name(self, class_name: str) -> Optional[str]:
        """Resolve a class name to its fully qualified name."""
        # Check cache first
        if class_name in self._class_cache:
            return self._class_cache[class_name]

        # If it's already a fully qualified name
        for m_name, analyzer in self.all_modules.items():
            if class_name in analyzer.classes:
                self._class_cache[class_name] = class_name
                return class_name

        # Check for imported classes
        if class_name in self.module_analyzer.imports.import_map:
            imported_path = self.module_analyzer.imports.import_map[class_name]
            self._class_cache[class_name] = imported_path
            return imported_path

        # Check if it's a local class in the current module
        local_class = f"{self.module_name}.{class_name}"
        if local_class in self.module_analyzer.classes:
            self._class_cache[class_name] = local_class
            return local_class

        # Try to find in star imports
        for star_module in self.module_analyzer.imports.star_imports:
            for m_name, analyzer in self.all_modules.items():
                if m_name == star_module or m_name.endswith('.' + star_module):
                    potential_class = f"{m_name}.{class_name}"
                    if potential_class in analyzer.classes:
                        self._class_cache[class_name] = potential_class
                        return potential_class

        # Not found
        self._class_cache[class_name] = None
        return None

    def _extract_call_target(self, call_node: ast.Call) -> Optional[str]:
        """Extract the target class or function being called."""
        if isinstance(call_node.func, ast.Name):
            # Direct call: ClassName()
            target_name = call_node.func.id

            # Check if it's an imported class
            if target_name in self.module_analyzer.imports.import_map:
                return self.module_analyzer.imports.import_map[target_name]

            # It might be a local class
            local_class = f"{self.module_name}.{target_name}"
            if local_class in self.module_analyzer.classes:
                return local_class

            # Otherwise, just return the name
            return target_name

        elif isinstance(call_node.func, ast.Attribute):
            # Qualified call: module.ClassName()
            parts = self._extract_attribute_chain(call_node.func)
            return '.'.join(parts) if parts else None

        return None

    def _extract_attribute_chain(self, node: ast.AST) -> List[str]:
        """Extract a chain of attribute access like module.submodule.function."""
        parts = []

        # Handle the base case of a Name node
        if isinstance(node, ast.Name):
            return [node.id]

        # Handle Attribute nodes recursively
        elif isinstance(node, ast.Attribute):
            # Get the value parts recursively
            value_parts = self._extract_attribute_chain(node.value)
            # Add the attribute
            return value_parts + [node.attr]

        return parts

    def _process_annotation(self, node: ast.AST) -> Optional[str]:
        """Process a type annotation and return a class name if possible."""
        if isinstance(node, ast.Name):
            # Simple annotation: MyClass
            return node.id
        elif isinstance(node, ast.Attribute):
            # Qualified annotation: module.MyClass
            parts = self._extract_attribute_chain(node)
            return '.'.join(parts) if parts else None
        elif isinstance(node, ast.Subscript):
            # Generic type: List[MyClass]
            # Just return the container type for now
            if isinstance(node.value, ast.Name):
                return node.value.id
            elif isinstance(node.value, ast.Attribute):
                parts = self._extract_attribute_chain(node.value)
                return '.'.join(parts) if parts else None
        return None

    def record_call(self, caller: str, callee: str, lineno: int):
        """Record a function call."""
        self.calls.append({
            'caller': caller,
            'callee': callee,
            'lineno': lineno
        })


@lru_cache(maxsize=128)
def parse_python_file(file_path: str) -> Optional[ast.AST]:
    """Parse a Python file and return its AST."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return ast.parse(file.read(), filename=file_path)
    except SyntaxError as e:
        logger.error(f"Syntax error in file: {file_path} - {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {str(e)}")
        return None


def find_project_python_files(project_path: str) -> List[str]:
    """Find all Python files within the project directory only."""
    python_files = []

    # Convert to absolute path for comparison
    abs_project_path = os.path.abspath(project_path)

    # Common directories to exclude
    exclude_patterns = [
        r'\.git', r'\.svn', r'\.hg',                     # Version control
        r'__pycache__', r'\.pytest_cache', r'\.mypy_cache', # Python caches
        r'venv', r'env', r'virtualenv', r'\.venv',       # Virtual environments
        r'node_modules', r'bower_components',            # JS dependencies
        r'\.egg-info', r'\.eggs', r'dist', r'build',     # Python packaging
        r'\.tox', r'\.coverage', r'htmlcov',             # Testing
    ]
    exclude_regex = re.compile('|'.join(f'({pattern})' for pattern in exclude_patterns))

    if os.path.isfile(abs_project_path) and abs_project_path.endswith('.py'):
        python_files = [abs_project_path]
    elif os.path.isdir(abs_project_path):
        for root, dirs, files in os.walk(abs_project_path):
            # Skip directories matching exclude patterns
            dirs[:] = [d for d in dirs if not exclude_regex.search(d)]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    python_files.append(file_path)

    logger.info(f"Found {len(python_files)} Python files in project")
    return python_files


def get_module_name(file_path: str, project_root: str) -> str:
    """Get the module name from a file path relative to project root."""
    # Get relative path from project root
    rel_path = os.path.relpath(file_path, project_root)

    # Convert path to module name
    module_parts = []
    current_path = os.path.dirname(rel_path)

    # Add file name without extension
    module_parts.insert(0, os.path.splitext(os.path.basename(rel_path))[0])

    # Add package hierarchy
    while current_path and current_path != '.':
        # Check for both regular packages with __init__.py and namespace packages (PEP 420)
        init_path = os.path.join(project_root, current_path, '__init__.py')
        is_namespace = any(
            os.path.isdir(os.path.join(project_root, current_path, d))
            and os.path.exists(os.path.join(project_root, current_path, d, '__init__.py'))
            for d in os.listdir(os.path.join(project_root, current_path))
            if os.path.isdir(os.path.join(project_root, current_path, d))
        )

        if os.path.isfile(init_path) or is_namespace:
            module_parts.insert(0, os.path.basename(current_path))

        current_path = os.path.dirname(current_path)

    return '.'.join(module_parts)


def analyze_project(py_files: List[str], project_root: str) -> Tuple[Dict[str, ModuleAnalyzer], List[Dict]]:
    """Analyze all modules in the project and extract function calls."""
    # First pass: analyze modules and collect class/function definitions
    module_analyzers = {}
    all_module_names = set()

    # Process files in parallel for better performance
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
        # First, parse all files
        future_to_file = {
            executor.submit(parse_python_file, file_path): (file_path, get_module_name(file_path, project_root))
            for file_path in py_files
        }

        # Collect results
        for future in concurrent.futures.as_completed(future_to_file):
            file_path, module_name = future_to_file[future]
            tree = future.result()
            all_module_names.add(module_name)

            if tree:
                logger.debug(f"Parsed module: {module_name}")
                module_analyzers[module_name] = ModuleAnalyzer(module_name, file_path, tree, project_root)

    # Second pass: analyze function calls (needs to be sequential due to dependencies)
    all_calls = []

    for module_name, analyzer in module_analyzers.items():
        logger.debug(f"Analyzing function calls in: {module_name}")
        visitor = FunctionCallVisitor(module_name, analyzer.file_path, analyzer, module_analyzers, all_module_names)
        visitor.visit(analyzer.tree)
        all_calls.extend(visitor.calls)

    return module_analyzers, all_calls


def build_call_graph(module_analyzers: Dict[str, ModuleAnalyzer], all_calls: List[Dict]) -> nx.DiGraph:
    """Build a call graph from the analyzed modules and calls."""
    G = nx.DiGraph()

    # Add nodes for all functions
    for module_name, analyzer in module_analyzers.items():
        for func_name, func_info in analyzer.functions.items():
            G.add_node(func_name, **{
                'name': func_info['name'],
                'module': module_name,
                'class': func_info.get('class'),
                'lineno': func_info.get('lineno', 0),
                'path': analyzer.file_path,
                'is_async': func_info.get('is_async', False),
                'is_property': func_info.get('is_property', False),
                'decorators': func_info.get('decorators', [])
            })

    # Build a lookup table for resolving function names
    function_lookup = {}
    for module_name, analyzer in module_analyzers.items():
        for func_name in analyzer.functions:
            # Extract the short name (without module/class)
            short_name = func_name.split('.')[-1]
            if short_name not in function_lookup:
                function_lookup[short_name] = []
            function_lookup[short_name].append(func_name)

    # Add edges for function calls
    for call in all_calls:
        caller = call['caller']
        callee = call['callee']
        lineno = call['lineno']

        # If caller and callee are both in the graph, add the edge directly
        if caller in G.nodes and callee in G.nodes:
            G.add_edge(caller, callee, lineno=lineno)
            continue

        # Try more advanced resolution
        resolved_callee = _resolve_function_call(caller, callee, G, function_lookup, module_analyzers)
        if resolved_callee:
            G.add_edge(caller, resolved_callee, lineno=lineno)

    # Handle cycles by marking edges that form cycles
    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            for i in range(len(cycle)):
                source = cycle[i]
                target = cycle[(i + 1) % len(cycle)]
                if G.has_edge(source, target):
                    G.edges[source, target]['is_cycle'] = True
    except Exception as e:
        logger.warning(f"Could not detect cycles: {str(e)}")

    return G


def _resolve_function_call(caller: str, callee: str, G: nx.DiGraph,
                          function_lookup: Dict[str, List[str]],
                          module_analyzers: Dict[str, ModuleAnalyzer]) -> Optional[str]:
    """Resolve a function call using advanced heuristics."""
    # If caller is in graph but callee needs resolution
    if caller in G.nodes:
        # Check if callee is a fully qualified name that might match with a prefix
        for node in G.nodes:
            if node.endswith('.' + callee) or node == callee:
                return node

        # Try to resolve by short name
        short_name = callee.split('.')[-1]
        if short_name in function_lookup:
            # If there's only one function with this name, use it
            if len(function_lookup[short_name]) == 1:
                return function_lookup[short_name][0]

            # Multiple options - try to find the best match
            # First try classes in the same module or package
            caller_parts = caller.split('.')
            caller_module = caller_parts[0]

            # If caller is a method, get its class
            caller_class = None
            if len(caller_parts) >= 3:  # module.class.method
                caller_class = '.'.join(caller_parts[:-1])

            # 1. Try to find methods in the same class
            if caller_class:
                for candidate in function_lookup[short_name]:
                    candidate_parts = candidate.split('.')
                    candidate_class = '.'.join(candidate_parts[:-1])
                    if candidate_class == caller_class:
                        return candidate

            # 2. Try to find functions in the same module
            for candidate in function_lookup[short_name]:
                candidate_parts = candidate.split('.')
                candidate_module = candidate_parts[0]
                if candidate_module == caller_module:
                    return candidate

            # 3. Check if the class part of callee matches the class part of caller
            if '.' in callee and caller_class:
                callee_parts = callee.split('.')
                if len(callee_parts) >= 2:
                    callee_class = callee_parts[0]
                    for candidate in function_lookup[short_name]:
                        candidate_parts = candidate.split('.')
                        if len(candidate_parts) >= 3:  # module.class.method
                            candidate_class = candidate_parts[1]
                            if candidate_class == callee_class:
                                return candidate

            # 4. If all else fails, just use the first one
            # This is not ideal but better than missing connections
            return function_lookup[short_name][0]

    return None


def filter_by_modules(G: nx.DiGraph, included_modules: List[str]) -> nx.DiGraph:
    """Filter graph to only include specified modules."""
    H = G.copy()
    nodes_to_remove = []

    for node in H.nodes():
        module = H.nodes[node].get('module', '')
        if not any(module.startswith(m) for m in included_modules):
            nodes_to_remove.append(node)

    H.remove_nodes_from(nodes_to_remove)
    return H


def filter_by_depth(G: nx.DiGraph, root_function: str, max_depth: int = 2) -> nx.DiGraph:
    """Filter graph to only include functions within a certain call depth."""
    H = nx.DiGraph()

    # Find the root node (might be a partial match)
    root_node = None
    for node in G.nodes():
        # Check for exact match first
        if node == root_function:
            root_node = node
            break

        # Then try partial match
        if root_function in node:
            root_node = node
            break

    if not root_node:
        logger.warning(f"Root function '{root_function}' not found in the graph")
        return H

    # Use BFS to find all nodes within max_depth
    visited = {root_node: 0}  # Node: depth
    queue = [(root_node, 0)]

    while queue:
        node, depth = queue.pop(0)
        if node in G.nodes():
            H.add_node(node, **G.nodes[node])

            if depth < max_depth:
                # Add outgoing edges (function calls)
                for _, neighbor in G.out_edges(node):
                    edge_data = G.get_edge_data(node, neighbor, {})
                    if neighbor not in visited or visited[neighbor] > depth + 1:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, depth + 1))
                        H.add_edge(node, neighbor, **edge_data)

    return H


def generate_styled_mermaid(G: nx.DiGraph) -> str:
    """Generate a beautifully styled Mermaid flowchart with vibrant colors and icons."""
    # Start with Mermaid header
    mermaid_code = [
        "flowchart LR",
        "    %% Node definitions with styling"
    ]

    # Define a vibrant color palette
    colors = {
        'module': {
            'primary': '#5D2E8C',  # Deep purple
            'secondary': '#7B4BAF'  # Medium purple
        },
        'class': {
            'primary': '#2962FF',  # Vibrant blue
            'secondary': '#5C8AFF'  # Lighter blue
        },
        'constructor': {
            'primary': '#E53935',  # Vibrant red
            'secondary': '#EF5350'  # Lighter red
        },
        'method': {
            'primary': '#00C853',  # Vibrant green
            'secondary': '#4CD964'  # Lighter green
        },
        'async': {
            'primary': '#AA00FF',  # Deep magenta
            'secondary': '#CE93D8'  # Lighter magenta
        },
        'property': {
            'primary': '#FF6D00',  # Deep orange
            'secondary': '#FFAB40'  # Lighter orange
        },
        'static': {
            'primary': '#00B0FF',  # Light blue
            'secondary': '#80D8FF'  # Lighter blue
        },
        'private': {
            'primary': '#757575',  # Dark gray
            'secondary': '#BDBDBD'  # Light gray
        }
    }

    # Track node IDs to ensure uniqueness
    node_ids = {}
    node_count = 0

    # Group nodes by module
    modules = {}
    for node in G.nodes():
        module = G.nodes[node].get('module', 'unknown')
        if module not in modules:
            modules[module] = []
        modules[module].append(node)

    # Create a title node
    mermaid_code.append(f"    title(\"fa:fa-project-diagram Python Method Visualization\"):::title")

    # Sort modules by name for consistent output
    sorted_modules = sorted(modules.items())

    # Process each module
    for module_idx, (module, nodes) in enumerate(sorted_modules):
        module_short_name = module.split('.')[-1]
        module_id = f"mod{module_idx}"

        # Add module subgraph with icon
        mermaid_code.append(f"    subgraph {module_id}[\"fa:fa-cube {module_short_name}\"]")
        mermaid_code.append(f"        direction TB")

        # Group nodes by class
        classes = {}
        standalone_nodes = []

        for node in nodes:
            class_name = G.nodes[node].get('class')
            if class_name:
                if class_name not in classes:
                    classes[class_name] = []
                classes[class_name].append(node)
            else:
                standalone_nodes.append(node)

        # Process each class
        for class_idx, (class_name, class_nodes) in enumerate(sorted(classes.items())):
            class_short_name = class_name.split('.')[-1]
            class_id = f"cls{module_idx}_{class_idx}"

            # Add class subgraph with icon
            mermaid_code.append(f"        subgraph {class_id}[\"fa:fa-code {class_short_name}\"]")
            mermaid_code.append(f"            direction TB")

            # Sort methods by type
            init_methods = []
            property_methods = []
            static_methods = []
            async_methods = []
            private_methods = []
            regular_methods = []

            for node in class_nodes:
                method_name = node.split('.')[-1]
                is_private = method_name.startswith('_') and not method_name.startswith(
                    '__') and not method_name.endswith('__')

                if method_name.startswith('__init__') or method_name.startswith('__new__'):
                    init_methods.append(node)
                elif G.nodes[node].get('is_property', False):
                    property_methods.append(node)
                elif any(d.get('name') == 'staticmethod' for d in G.nodes[node].get('decorators', [])):
                    static_methods.append(node)
                elif G.nodes[node].get('is_async', False):
                    async_methods.append(node)
                elif is_private:
                    private_methods.append(node)
                else:
                    regular_methods.append(node)

            # Process methods in order of importance
            for node_list, icon, style_class in [
                (init_methods, "fa:fa-play-circle", "constructor"),
                (property_methods, "fa:fa-lock", "property"),
                (static_methods, "fa:fa-cog", "static"),
                (async_methods, "fa:fa-bolt", "async"),
                (regular_methods, "fa:fa-code-branch", "method"),
                (private_methods, "fa:fa-key", "private")
            ]:
                for node in node_list:
                    method_name = node.split('.')[-1]
                    # Generate a unique ID for this node
                    if node in node_ids:
                        node_id = node_ids[node]
                    else:
                        node_id = f"node{node_count}"
                        node_ids[node] = node_id
                        node_count += 1

                    # Add the node with its icon and style
                    mermaid_code.append(f"                {node_id}[\"{icon} {method_name}\"]:::{style_class}")

            mermaid_code.append("        end")

        # Process standalone functions
        if standalone_nodes:
            func_id = f"func{module_idx}"
            mermaid_code.append(f"        subgraph {func_id}[\"fa:fa-sitemap Module Functions\"]")

            for node in standalone_nodes:
                func_name = node.split('.')[-1]
                is_private = func_name.startswith('_') and not func_name.startswith('__') and not func_name.endswith(
                    '__')

                # Generate a unique ID for this node
                if node in node_ids:
                    node_id = node_ids[node]
                else:
                    node_id = f"node{node_count}"
                    node_ids[node] = node_id
                    node_count += 1

                # Determine icon and style based on function type
                if G.nodes[node].get('is_async', False):
                    mermaid_code.append(f"            {node_id}[\"fa:fa-bolt {func_name}\"]:::async")
                elif is_private:
                    mermaid_code.append(f"            {node_id}[\"fa:fa-key {func_name}\"]:::private")
                elif G.nodes[node].get('decorators', []):
                    mermaid_code.append(f"            {node_id}[\"fa:fa-star {func_name}\"]:::decorated")
                else:
                    mermaid_code.append(f"            {node_id}[\"fa:fa-code-branch {func_name}\"]:::method")

            mermaid_code.append("        end")

        mermaid_code.append("    end")

    # Add a legend section
    mermaid_code.append("    subgraph legend[\"Legend\"]")
    mermaid_code.append("        l1[\"fa:fa-play-circle Constructor\"]:::constructor")
    mermaid_code.append("        l2[\"fa:fa-code-branch Method\"]:::method")
    mermaid_code.append("        l3[\"fa:fa-bolt Async Method\"]:::async")
    mermaid_code.append("        l4[\"fa:fa-lock Property\"]:::property")
    mermaid_code.append("        l5[\"fa:fa-cog Static Method\"]:::static")
    mermaid_code.append("        l6[\"fa:fa-key Private Method\"]:::private")
    mermaid_code.append("        l7[\"fa:fa-star Decorated Method\"]:::decorated")
    mermaid_code.append("    end")

    # Add connections between nodes
    mermaid_code.append("")
    mermaid_code.append("    %% Connections between methods")

    # Process edges, using the node IDs we created
    for source, target, data in G.edges(data=True):
        if source in node_ids and target in node_ids:
            source_id = node_ids[source]
            target_id = node_ids[target]

            # Check if this edge is part of a cycle
            if data.get('is_cycle'):
                mermaid_code.append(f"    {source_id} -.-> {target_id}")
            # Check if it's a callback relationship
            elif any(kw in source.lower() for kw in ['callback', 'handler', 'listener']):
                mermaid_code.append(f"    {source_id} ==> {target_id}")
            # Check if it's likely a dependency injection
            elif target.lower().endswith(('factory', 'provider', 'service')):
                mermaid_code.append(f"    {source_id} --o {target_id}")
            else:
                # Regular call
                mermaid_code.append(f"    {source_id} --> {target_id}")

    # Add styling
    mermaid_code.append("")
    mermaid_code.append("    %% Styling")
    mermaid_code.append(
        f"    style title color:#ffffff, fill:{colors['module']['primary']}, stroke:{colors['module']['primary']}, stroke-width:0px, font-size:18px")
    mermaid_code.append(
        f"    classDef constructor color:#ffffff, fill:{colors['constructor']['primary']}, stroke:{colors['constructor']['secondary']}")
    mermaid_code.append(
        f"    classDef method color:#ffffff, fill:{colors['method']['primary']}, stroke:{colors['method']['secondary']}")
    mermaid_code.append(
        f"    classDef async color:#ffffff, fill:{colors['async']['primary']}, stroke:{colors['async']['secondary']}")
    mermaid_code.append(
        f"    classDef property color:#ffffff, fill:{colors['property']['primary']}, stroke:{colors['property']['secondary']}")
    mermaid_code.append(
        f"    classDef static color:#ffffff, fill:{colors['static']['primary']}, stroke:{colors['static']['secondary']}")
    mermaid_code.append(
        f"    classDef private color:#ffffff, fill:{colors['private']['primary']}, stroke:{colors['private']['secondary']}")
    mermaid_code.append(
        f"    classDef decorated color:#ffffff, fill:{colors['class']['primary']}, stroke:{colors['class']['secondary']}")

    return '\n'.join(mermaid_code)


def create_interactive_html(mermaid_code: str, project_name: str) -> str:
    """Create a beautiful interactive HTML page for the Mermaid diagram."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - Method Visualization</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        :root {{
            --primary-color: #5D2E8C;
            --secondary-color: #2962FF;
            --accent-color: #00C853;
            --background-color: #f8f9fa;
            --card-bg-color: #ffffff;
            --text-color: #333333;
            --border-radius: 8px;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        h1 {{
            margin: 0;
            font-size: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        h1 i {{
            margin-right: 10px;
            font-size: 24px;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }}

        .controls-container {{
            background-color: var(--card-bg-color);
            border-radius: var(--border-radius);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            padding: 20px;
            margin-bottom: 20px;
        }}

        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
            flex: 1;
            min-width: 200px;
        }}

        .control-group label {{
            font-weight: 600;
            margin-bottom: 5px;
            font-size: 14px;
            color: var(--text-color);
        }}

        .controls input,
        .controls select {{
            padding: 10px 15px;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            font-size: 14px;
            width: 100%;
        }}

        .button-group {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}

        button {{
            padding: 10px 15px;
            border: none;
            border-radius: var(--border-radius);
            background-color: var(--primary-color);
            color: white;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }}

        button:hover {{
            opacity: 0.9;
            transform: translateY(-2px);
        }}

        button i {{
            margin-right: 8px;
        }}

        button.secondary {{
            background-color: var(--secondary-color);
        }}

        button.accent {{
            background-color: var(--accent-color);
        }}

        .diagram-container {{
            background-color: var(--card-bg-color);
            border-radius: var(--border-radius);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            padding: 20px;
            overflow: auto;
            flex: 1;
            min-height: 600px;
            position: relative;
        }}

        .mermaid {{
            display: flex;
            justify-content: center;
        }}

        .zoom-controls {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            flex-direction: column;
        }}

        .zoom-controls button {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}

        .zoom-controls button i {{
            margin: 0;
        }}

        .theme-switch {{
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: var(--primary-color);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}

        footer {{
            background-color: var(--card-bg-color);
            padding: 15px;
            text-align: center;
            font-size: 14px;
            border-top: 1px solid #eee;
            margin-top: 20px;
        }}

        /* Dark theme */
        body.dark-mode {{
            --primary-color: #9C64FE;
            --secondary-color: #448AFF;
            --accent-color: #4CD964;
            --background-color: #121212;
            --card-bg-color: #1E1E1E;
            --text-color: #E0E0E0;
        }}

        /* Loading indicator */
        .loading {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 20px;
            z-index: 100;
        }}

        .loading i {{
            animation: spin 1s infinite linear;
            font-size: 48px;
        }}

        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="theme-switch" id="themeSwitch" title="Toggle dark mode">
        <i class="fas fa-moon"></i>
    </div>

    <header>
        <h1><i class="fas fa-project-diagram"></i> {project_name} Method Visualization</h1>
    </header>

    <div class="container">
        <div class="controls-container">
            <div class="controls">
                <div class="control-group">
                    <label for="searchInput"><i class="fas fa-search"></i> Search Methods</label>
                    <input type="text" id="searchInput" placeholder="Type to search methods...">
                </div>

                <div class="control-group">
                    <label for="moduleFilter"><i class="fas fa-filter"></i> Filter by Module</label>
                    <select id="moduleFilter">
                        <option value="">All Modules</option>
                    </select>
                </div>

                <div class="button-group">
                    <button id="expandAll"><i class="fas fa-expand-arrows-alt"></i> Expand All</button>
                    <button id="collapseAll" class="secondary"><i class="fas fa-compress-arrows-alt"></i> Collapse All</button>
                    <button id="downloadSVG" class="accent"><i class="fas fa-download"></i> Download SVG</button>
                </div>
            </div>
        </div>

        <div class="diagram-container">
            <div class="loading">
                <i class="fas fa-spinner"></i>
            </div>
            <div class="mermaid" id="mermaidGraph">
{mermaid_code}
            </div>
        </div>
    </div>

    <div class="zoom-controls">
        <button id="zoomIn" title="Zoom In"><i class="fas fa-plus"></i></button>
        <button id="zoomOut" title="Zoom Out"><i class="fas fa-minus"></i></button>
        <button id="resetZoom" title="Reset Zoom"><i class="fas fa-sync-alt"></i></button>
    </div>

    <footer>
        Generated by Python Method Visualization Tool
    </footer>

    <script>
        // Initialize Mermaid with advanced config
        mermaid.initialize({{
            startOnLoad: true,
            securityLevel: 'loose',
            theme: 'default',
            logLevel: 'error',
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});

        document.addEventListener('DOMContentLoaded', () => {{
            // Hide loading indicator when mermaid finishes rendering
            mermaid.contentLoaded();
            setTimeout(() => {{
                document.querySelector('.loading').style.display = 'none';
                initializeUI();
            }}, 1000);
        }});

        function initializeUI() {{
            const svgDocument = document.querySelector('.mermaid svg');
            const searchInput = document.getElementById('searchInput');
            const moduleSelect = document.getElementById('moduleFilter');
            const expandAll = document.getElementById('expandAll');
            const collapseAll = document.getElementById('collapseAll');
            const downloadSVG = document.getElementById('downloadSVG');
            const zoomIn = document.getElementById('zoomIn');
            const zoomOut = document.getElementById('zoomOut');
            const resetZoom = document.getElementById('resetZoom');
            const themeSwitch = document.getElementById('themeSwitch');

            if (!svgDocument) return;

            // Populate module filter
            const modules = Array.from(svgDocument.querySelectorAll('g.cluster[id^="flowchart-mod"]'))
                .map(g => {{
                    const label = g.querySelector('.nodeLabel');
                    return label ? label.textContent.trim().replace('fa:fa-cube ', '') : '';
                }})
                .filter(Boolean);

            modules.forEach(module => {{
                const option = document.createElement('option');
                option.value = module;
                option.textContent = module;
                moduleSelect.appendChild(option);
            }});

            // Search functionality
            searchInput.addEventListener('input', () => {{
                const searchTerm = searchInput.value.toLowerCase();
                highlightMatchingNodes(searchTerm);
            }});

            function highlightMatchingNodes(searchTerm) {{
                const allNodes = svgDocument.querySelectorAll('.node');

                allNodes.forEach(node => {{
                    const label = node.querySelector('.nodeLabel');
                    const text = label ? label.textContent.trim() : '';
                    const match = text.toLowerCase().includes(searchTerm);

                    // Clear existing highlights and styles
                    node.classList.remove('highlight');
                    node.style.opacity = '';

                    if (searchTerm && !match) {{
                        node.style.opacity = '0.3';
                    }} else if (searchTerm && match) {{
                        node.style.opacity = '1';
                        // Add a subtle animation effect
                        node.style.filter = 'drop-shadow(0 0 8px var(--primary-color))';
                    }} else {{
                        node.style.filter = '';
                    }}
                }});
            }}

            // Module filtering
            moduleSelect.addEventListener('change', () => {{
                const selectedModule = moduleSelect.value;
                filterByModule(selectedModule);
            }});

            function filterByModule(moduleName) {{
                const clusters = svgDocument.querySelectorAll('g.cluster[id^="flowchart-mod"]');

                clusters.forEach(cluster => {{
                    const label = cluster.querySelector('.nodeLabel');
                    const clusterModule = label ? label.textContent.trim().replace('fa:fa-cube ', '') : '';

                    if (!moduleName || clusterModule === moduleName) {{
                        cluster.style.opacity = '1';
                    }} else {{
                        cluster.style.opacity = '0.3';
                    }}
                }});
            }}

            // Expand/collapse functionality
            expandAll.addEventListener('click', () => {{
                const clusters = svgDocument.querySelectorAll('g.cluster');
                clusters.forEach(cluster => {{
                    cluster.style.opacity = '1';
                    cluster.style.display = 'block';
                }});
            }});

            collapseAll.addEventListener('click', () => {{
                // Keep module clusters visible but collapse class clusters
                const classClusters = svgDocument.querySelectorAll('g.cluster[id^="flowchart-cls"], g.cluster[id^="flowchart-func"]');
                classClusters.forEach(cluster => {{
                    cluster.style.display = 'none';
                }});
            }});

            // SVG download
            downloadSVG.addEventListener('click', () => {{
                // Create a clean copy of the SVG
                const svgCopy = svgDocument.cloneNode(true);

                // Set explicit width and height if needed
                svgCopy.setAttribute('width', svgDocument.getBBox().width);
                svgCopy.setAttribute('height', svgDocument.getBBox().height);

                // Convert to a data URL
                const svgData = new XMLSerializer().serializeToString(svgCopy);
                const svgBlob = new Blob([svgData], {{type: 'image/svg+xml;charset=utf-8'}});
                const svgUrl = URL.createObjectURL(svgBlob);

                // Create and trigger download
                const downloadLink = document.createElement('a');
                downloadLink.href = svgUrl;
                downloadLink.download = '{project_name}_methods.svg';
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
                URL.revokeObjectURL(svgUrl);
            }});

            // Zoom controls
            let zoomLevel = 1;
            const svgContainer = svgDocument.parentElement;

            zoomIn.addEventListener('click', () => {{
                zoomLevel = Math.min(zoomLevel * 1.2, 5); // Max zoom: 5x
                applyZoom();
            }});

            zoomOut.addEventListener('click', () => {{
                zoomLevel = Math.max(zoomLevel / 1.2, 0.2); // Min zoom: 0.2x
                applyZoom();
            }});

            resetZoom.addEventListener('click', () => {{
                zoomLevel = 1;
                applyZoom();
            }});

            function applyZoom() {{
                svgDocument.style.transform = `scale(${{zoomLevel}})`;
                svgDocument.style.transformOrigin = 'top left';
            }}

            // Theme toggle
            let darkMode = false;

            themeSwitch.addEventListener('click', () => {{
                darkMode = !darkMode;
                document.body.classList.toggle('dark-mode', darkMode);
                themeSwitch.innerHTML = darkMode 
                    ? '<i class="fas fa-sun"></i>' 
                    : '<i class="fas fa-moon"></i>';
            }});
        }}
    </script>
</body>
</html>
"""
    return html


def export_diagram(mermaid_code: str, output_path: str, output_format: str = 'mermaid', project_name: str = "Project"):
    """Export the Mermaid diagram to the specified format with enhanced visuals."""
    if output_format == 'mermaid':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        logger.info(f"Mermaid diagram saved to {output_path}")

        # Create an enhanced HTML version
        html_path = f"{os.path.splitext(output_path)[0]}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(create_interactive_html(mermaid_code, project_name))
        logger.info(f"Interactive HTML diagram saved to {html_path}")
    else:
        try:
            # Check if node and mmdc are available
            import subprocess

            # Save the mermaid code to a temporary file
            temp_file = f"{output_path}.tmp.mmd"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)

            # Run mmdc with enhanced configuration for better output
            # Add puppeteer arguments for better rendering
            result = subprocess.run(
                [
                    'mmdc',
                    '-i', temp_file,
                    '-o', output_path,
                    '-b', 'transparent',
                    '-w', '2000',  # Wider output
                    '-H', '1500',  # Taller output
                    '-p', '{"puppeteerConfig": {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}}'
                ],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                logger.error(f"Error generating {output_format}: {result.stderr}")
                logger.info("Falling back to mermaid format")
                # Save as mermaid format instead
                with open(f"{output_path}.mmd", 'w', encoding='utf-8') as f:
                    f.write(mermaid_code)
            else:
                logger.info(f"{output_format.upper()} diagram saved to {output_path}")

            # Always create an HTML version for better interactivity
            html_path = f"{os.path.splitext(output_path)[0]}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(create_interactive_html(mermaid_code, project_name))
            logger.info(f"Interactive HTML diagram saved to {html_path}")

            # Clean up the temporary file
            try:
                os.remove(temp_file)
            except Exception:
                pass

        except (ImportError, FileNotFoundError):
            logger.error(f"Could not generate {output_format}. Make sure Node.js and mermaid-cli are installed.")
            logger.info("Saving as mermaid format instead")
            # Save as mermaid format instead
            with open(f"{output_path}.mmd", 'w', encoding='utf-8') as f:
                f.write(mermaid_code)

            # Create HTML anyway
            html_path = f"{os.path.splitext(output_path)[0]}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(create_interactive_html(mermaid_code, project_name))
            logger.info(f"Interactive HTML diagram saved to {html_path}")


def create_d3_html_template(graph_data: dict, project_name: str) -> str:
    """Create an HTML template with D3.js visualization."""
    # Convert graph data to JSON string
    import json
    graph_json = json.dumps(graph_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - Code Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {{
            --primary-color: #5D2E8C;
            --secondary-color: #2962FF;
            --accent-color: #00C853;
            --danger-color: #F44336;
            --warning-color: #FF9800;
            --text-color: #212121;
            --text-secondary: #757575;
            --background-color: #f8f9fa;
            --card-bg-color: #ffffff;
            --border-color: #e0e0e0;
            --shadow-color: rgba(0,0,0,0.1);
            --node-constructor: #E53935;
            --node-method: #2962FF;
            --node-function: #00C853;
            --node-property: #FF6D00;
            --node-async: #AA00FF;
            --node-private: #757575;
            --node-cycle: #F44336;
            --link-normal: #616161;
            --link-cycle: #F44336;
        }}

        body.dark-mode {{
            --primary-color: #9C64FE;
            --secondary-color: #448AFF;
            --accent-color: #4CD964;
            --danger-color: #FF5252;
            --warning-color: #FFAB40;
            --text-color: #EEEEEE;
            --text-secondary: #BDBDBD;
            --background-color: #121212;
            --card-bg-color: #1E1E1E;
            --border-color: #333333;
            --shadow-color: rgba(0,0,0,0.5);
            --node-constructor: #FF5252;
            --node-method: #448AFF;
            --node-function: #4CD964;
            --node-property: #FFAB40;
            --node-async: #CE93D8;
            --node-private: #BDBDBD;
            --node-cycle: #FF5252;
            --link-normal: #BDBDBD;
            --link-cycle: #FF5252;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            transition: background-color 0.3s ease;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 5px var(--shadow-color);
            position: relative;
            z-index: 10;
        }}

        .project-title {{
            display: flex;
            align-items: center;
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .project-title i {{
            margin-right: 0.5rem;
            font-size: 1.25rem;
        }}

        .header-controls {{
            display: flex;
            gap: 0.75rem;
        }}

        .theme-toggle {{
            background: rgba(255, 255, 255, 0.2);
            border: none;
            border-radius: 50%;
            width: 2.5rem;
            height: 2.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s;
        }}

        .theme-toggle:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}

        .layout-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        .sidebar {{
            width: 300px;
            background-color: var(--card-bg-color);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease;
            z-index: 5;
            overflow-y: auto;
        }}

        .sidebar.collapsed {{
            transform: translateX(-100%);
        }}

        .sidebar-toggle {{
            position: absolute;
            left: 300px;
            top: 5rem;
            background: var(--card-bg-color);
            border: 1px solid var(--border-color);
            border-left: none;
            border-radius: 0 4px 4px 0;
            padding: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 10;
            box-shadow: 2px 0 5px var(--shadow-color);
            transition: left 0.3s ease;
        }}

        .sidebar-toggle.collapsed {{
            left: 0;
        }}

        .controls-section {{
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .controls-section h3 {{
            font-size: 1rem;
            margin-bottom: 0.75rem;
            color: var(--text-color);
            display: flex;
            align-items: center;
        }}

        .controls-section h3 i {{
            margin-right: 0.5rem;
        }}

        .search-box {{
            position: relative;
            margin-bottom: 1rem;
        }}

        .search-box input {{
            width: 100%;
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-size: 0.9rem;
            color: var(--text-color);
            background-color: var(--card-bg-color);
            transition: border-color 0.3s;
        }}

        .search-box i {{
            position: absolute;
            left: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
        }}

        select {{
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-size: 0.9rem;
            color: var(--text-color);
            background-color: var(--card-bg-color);
            margin-bottom: 1rem;
        }}

        .filter-label {{
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .btn-group {{
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }}

        button {{
            padding: 0.6rem 1rem;
            border: none;
            border-radius: 4px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }}

        button i {{
            margin-right: 0.5rem;
        }}

        .btn-primary {{
            background-color: var(--primary-color);
            color: white;
        }}

        .btn-primary:hover {{
            background-color: var(--secondary-color);
        }}

        .btn-secondary {{
            background-color: var(--card-bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-color);
        }}

        .btn-secondary:hover {{
            background-color: var(--border-color);
        }}

        .metrics-section {{
            padding: 1rem;
        }}

        .metric-card {{
            background-color: var(--card-bg-color);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px var(--shadow-color);
        }}

        .metric-title {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}

        .metric-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--primary-color);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }}

        .cycles-section {{
            padding: 1rem;
            max-height: 300px;
            overflow-y: auto;
        }}

        .cycle-item {{
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            margin-bottom: 0.75rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }}

        .cycle-item:hover {{
            background-color: rgba(var(--primary-color-rgb), 0.1);
        }}

        .cycle-header {{
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        }}

        .cycle-id {{
            background-color: var(--danger-color);
            color: white;
            border-radius: 50%;
            width: 1.5rem;
            height: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            margin-right: 0.75rem;
        }}

        .cycle-length {{
            margin-left: auto;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .cycle-description {{
            font-size: 0.9rem;
            word-break: break-word;
        }}

        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }}

        .toolbar {{
            padding: 0.75rem;
            background-color: var(--card-bg-color);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .zoom-controls {{
            display: flex;
            gap: 0.5rem;
        }}

        .zoom-controls button {{
            width: 2.25rem;
            height: 2.25rem;
            border-radius: 50%;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .zoom-controls button i {{
            margin: 0;
        }}

        .viz-controls {{
            display: flex;
            gap: 0.5rem;
        }}

        #graph-container {{
            flex: 1;
            overflow: hidden;
            position: relative;
        }}

        svg {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        .tooltip {{
            position: absolute;
            padding: 0.75rem;
            background-color: var(--card-bg-color);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            box-shadow: 0 2px 10px var(--shadow-color);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 300px;
            z-index: 1000;
        }}

        .tooltip h4 {{
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }}

        .tooltip p {{
            margin-bottom: 0.25rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .tooltip .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
            margin-right: 0.25rem;
        }}

        .badge-constructor {{
            background-color: var(--node-constructor);
        }}

        .badge-method {{
            background-color: var(--node-method);
        }}

        .badge-function {{
            background-color: var(--node-function);
        }}

        .badge-property {{
            background-color: var(--node-property);
        }}

        .badge-async {{
            background-color: var(--node-async);
        }}

        .badge-private {{
            background-color: var(--node-private);
        }}

        .badge-cycle {{
            background-color: var(--node-cycle);
        }}

        .node {{
            cursor: pointer;
            transition: opacity 0.3s;
        }}

        .node circle, .node rect {{
            transition: all 0.3s;
        }}

        .node text {{
            dominant-baseline: central;
            text-anchor: middle;
            font-size: 10px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            pointer-events: none;
            transition: all 0.3s;
        }}

        .node.faded {{
            opacity: 0.2;
        }}

        .node.highlighted circle, .node.highlighted rect {{
            stroke-width: 3px;
            stroke: #FFF;
            filter: drop-shadow(0 0 3px rgba(0,0,0,0.3));
        }}

        .link {{
            stroke-opacity: 0.6;
            transition: stroke-opacity 0.3s, stroke-width 0.3s;
        }}

        .link.faded {{
            stroke-opacity: 0.1;
        }}

        .link.highlighted {{
            stroke-opacity: 1;
            stroke-width: 2.5px;
        }}

        .legend {{
            position: absolute;
            bottom: 1rem;
            right: 1rem;
            background-color: var(--card-bg-color);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 0.75rem;
            box-shadow: 0 2px 5px var(--shadow-color);
            z-index: 5;
        }}

        .legend-title {{
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}

        .legend-items {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 0.8rem;
        }}

        .legend-color {{
            width: 1rem;
            height: 1rem;
            border-radius: 3px;
            margin-right: 0.5rem;
        }}

        .loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.6);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            color: white;
            flex-direction: column;
        }}

        .loading-spinner {{
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid white;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin-bottom: 1rem;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        @media (max-width: 768px) {{
            .sidebar {{
                position: absolute;
                height: 100%;
                transform: translateX(-100%);
            }}

            .sidebar.active {{
                transform: translateX(0);
            }}

            .sidebar-toggle {{
                left: 0;
            }}

            .sidebar-toggle.active {{
                left: 300px;
            }}

            .metrics-grid {{
                grid-template-columns: 1fr;
            }}

            .toolbar {{
                flex-wrap: wrap;
            }}

            .zoom-controls, .viz-controls {{
                margin-bottom: 0.5rem;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="project-title">
            <i class="fas fa-project-diagram"></i>
            {project_name}
        </div>
        <div class="header-controls">
            <button class="theme-toggle" aria-label="Toggle dark mode">
                <i class="fas fa-moon"></i>
            </button>
        </div>
    </header>

    <div class="layout-container">
        <aside class="sidebar">
            <div class="controls-section">
                <h3><i class="fas fa-search"></i> Search & Filter</h3>
                <div class="search-box">
                    <i class="fas fa-search"></i>
                    <input type="text" id="search-input" placeholder="Search methods or classes...">
                </div>

                <label class="filter-label">Module Filter</label>
                <select id="module-filter">
                    <option value="">All Modules</option>
                    <!-- Will be populated by JS -->
                </select>

                <label class="filter-label">Node Type Filter</label>
                <select id="type-filter">
                    <option value="">All Types</option>
                    <option value="constructor">Constructors</option>
                    <option value="method">Methods</option>
                    <option value="function">Functions</option>
                    <option value="property">Properties</option>
                    <option value="async">Async Methods</option>
                    <option value="private">Private Methods</option>
                </select>

                <div class="btn-group">
                    <button id="highlight-cycles" class="btn-primary">
                        <i class="fas fa-sync-alt"></i> Highlight Cycles
                    </button>
                    <button id="reset-filters" class="btn-secondary">
                        <i class="fas fa-undo"></i> Reset
                    </button>
                </div>
            </div>

            <div class="metrics-section">
                <h3><i class="fas fa-chart-bar"></i> Project Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Total Methods</div>
                        <div class="metric-value" id="total-methods">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Total Modules</div>
                        <div class="metric-value" id="total-modules">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Relationships</div>
                        <div class="metric-value" id="total-relationships">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Cyclic Dependencies</div>
                        <div class="metric-value" id="total-cycles">-</div>
                    </div>
                </div>
            </div>

            <div class="cycles-section">
                <h3><i class="fas fa-exclamation-triangle"></i> Cyclic Dependencies</h3>
                <div id="cycles-list">
                    <!-- Will be populated by JS -->
                </div>
            </div>
        </aside>

        <div class="sidebar-toggle">
            <i class="fas fa-chevron-right"></i>
        </div>

        <main class="main-content">
            <div class="toolbar">
                <div class="zoom-controls">
                    <button id="zoom-in" class="btn-secondary" aria-label="Zoom in">
                        <i class="fas fa-plus"></i>
                    </button>
                    <button id="zoom-out" class="btn-secondary" aria-label="Zoom out">
                        <i class="fas fa-minus"></i>
                    </button>
                    <button id="zoom-reset" class="btn-secondary" aria-label="Reset zoom">
                        <i class="fas fa-expand"></i>
                    </button>
                </div>
                <div class="viz-controls">
                    <button id="export-svg" class="btn-primary">
                        <i class="fas fa-download"></i> Export SVG
                    </button>
                    <button id="toggle-force" class="btn-secondary">
                        <i class="fas fa-pause"></i> Pause Layout
                    </button>
                </div>
            </div>

            <div id="graph-container">
                <!-- D3 visualization will be rendered here -->
            </div>

            <div class="tooltip" id="node-tooltip"></div>

            <div class="legend">
                <div class="legend-title">Legend</div>
                <div class="legend-items">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color:var(--node-constructor)"></div>
                        <span>Constructor</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color:var(--node-method)"></div>
                        <span>Method</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color:var(--node-function)"></div>
                        <span>Function</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color:var(--node-property)"></div>
                        <span>Property</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color:var(--node-async)"></div>
                        <span>Async</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color:var(--node-private)"></div>
                        <span>Private</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color:var(--node-cycle)"></div>
                        <span>In Cycle</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color:var(--link-cycle)"></div>
                        <span>Cyclic Link</span>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <div class="loading-overlay" id="loading">
        <div class="loading-spinner"></div>
        <div>Building visualization...</div>
    </div>

    <script>
        // Store graph data from Python
        const graphData = {graph_json};

        // DOM elements
        const graphContainer = document.getElementById('graph-container');
        const tooltip = document.getElementById('node-tooltip');
        const searchInput = document.getElementById('search-input');
        const moduleFilter = document.getElementById('module-filter');
        const typeFilter = document.getElementById('type-filter');
        const highlightCyclesBtn = document.getElementById('highlight-cycles');
        const resetFiltersBtn = document.getElementById('reset-filters');
        const zoomInBtn = document.getElementById('zoom-in');
        const zoomOutBtn = document.getElementById('zoom-out');
        const zoomResetBtn = document.getElementById('zoom-reset');
        const exportSvgBtn = document.getElementById('export-svg');
        const toggleForceBtn = document.getElementById('toggle-force');
        const themeToggle = document.querySelector('.theme-toggle');
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');
        const cyclesList = document.getElementById('cycles-list');
        const loadingOverlay = document.getElementById('loading');

        // Populate metrics
        document.getElementById('total-methods').textContent = graphData.project_info.total_nodes;
        document.getElementById('total-modules').textContent = graphData.project_info.modules.length;
        document.getElementById('total-relationships').textContent = graphData.project_info.total_links;
        document.getElementById('total-cycles').textContent = graphData.project_info.total_cycles;

        // Populate module filter
        graphData.project_info.modules.sort().forEach(module => {{
            const option = document.createElement('option');
            option.value = module;
            option.textContent = module;
            moduleFilter.appendChild(option);
        }});

        // Populate cycles list
        if (graphData.cycles.length === 0) {{
            cyclesList.innerHTML = '<p style="color:var(--text-secondary);font-size:0.9rem;">No cyclic dependencies detected.</p>';
        }} else {{
            graphData.cycles.forEach(cycle => {{
                const cycleItem = document.createElement('div');
                cycleItem.className = 'cycle-item';
                cycleItem.dataset.nodes = JSON.stringify(cycle.nodes);

                cycleItem.innerHTML = `
                    <div class="cycle-header">
                        <div class="cycle-id">${{cycle.id + 1}}</div>
                        <div class="cycle-length">${{cycle.length}} nodes</div>
                    </div>
                    <div class="cycle-description">${{cycle.description}}</div>
                `;

                cyclesList.appendChild(cycleItem);
            }});
        }}

        // Theme toggling
        let darkMode = false;
        themeToggle.addEventListener('click', () => {{
            darkMode = !darkMode;
            document.body.classList.toggle('dark-mode', darkMode);
            themeToggle.innerHTML = darkMode 
                ? '<i class="fas fa-sun"></i>' 
                : '<i class="fas fa-moon"></i>';
        }});

        // Sidebar toggling
        let sidebarCollapsed = false;
        sidebarToggle.addEventListener('click', () => {{
            sidebarCollapsed = !sidebarCollapsed;
            sidebar.classList.toggle('collapsed', sidebarCollapsed);
            sidebarToggle.classList.toggle('collapsed', sidebarCollapsed);
            sidebarToggle.innerHTML = sidebarCollapsed 
                ? '<i class="fas fa-chevron-right"></i>' 
                : '<i class="fas fa-chevron-left"></i>';
        }});

        // D3.js visualization
        let svg, simulation, nodeElements, linkElements;
        let width, height;
        let zoom;
        let isDragging = false;

        // Set initial node positions using a radial layout
        const setInitialPositions = () => {{
            const centerX = width / 2;
            const centerY = height / 2;
            const radius = Math.min(width, height) * 0.35;

            // Group nodes by module
            const moduleGroups = {{}};
            graphData.nodes.forEach(node => {{
                if (!moduleGroups[node.module]) {{
                    moduleGroups[node.module] = [];
                }}
                moduleGroups[node.module].push(node);
            }});

            // Position nodes in a circle, grouped by module
            const moduleCount = Object.keys(moduleGroups).length;
            let moduleIndex = 0;

            for (const module in moduleGroups) {{
                const moduleAngle = (Math.PI * 2 * moduleIndex) / moduleCount;
                const moduleRadius = radius * 0.8;
                const moduleX = centerX + moduleRadius * Math.cos(moduleAngle);
                const moduleY = centerY + moduleRadius * Math.sin(moduleAngle);

                const nodes = moduleGroups[module];
                const nodeCount = nodes.length;

                nodes.forEach((node, i) => {{
                    const nodeAngle = moduleAngle + (Math.PI * 0.5 * (i - nodeCount / 2) / nodeCount);
                    const nodeRadius = radius * 0.4;
                    node.x = moduleX + nodeRadius * Math.cos(nodeAngle);
                    node.y = moduleY + nodeRadius * Math.sin(nodeAngle);
                }});

                moduleIndex++;
            }}
        }};

        // Initialize the visualization
        const initializeVisualization = () => {{
            width = graphContainer.clientWidth;
            height = graphContainer.clientHeight;

            // Create SVG element
            svg = d3.select('#graph-container')
                .append('svg')
                .attr('width', width)
                .attr('height', height)
                .attr('viewBox', [0, 0, width, height])
                .attr('style', 'max-width: 100%; height: auto;');

            // Create SVG groups for links and nodes with proper layering
            const g = svg.append('g');
            const linksGroup = g.append('g').attr('class', 'links');
            const nodesGroup = g.append('g').attr('class', 'nodes');

            // Setup zoom behavior
            zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {{
                    g.attr('transform', event.transform);
                }});

            svg.call(zoom);

            // Set initial positions
            setInitialPositions();

            // Create links
            linkElements = linksGroup.selectAll('line')
                .data(graphData.links)
                .enter()
                .append('line')
                .attr('class', d => `link ${{d.is_cycle ? 'cycle-link' : ''}}`)
                .attr('stroke', d => d.is_cycle ? 'var(--link-cycle)' : 'var(--link-normal)')
                .attr('stroke-width', d => d.is_cycle ? 2 : 1.5)
                .attr('marker-end', d => d.is_cycle ? 'url(#arrow-cycle)' : 'url(#arrow)');

            // Create nodes
            nodeElements = nodesGroup.selectAll('.node')
                .data(graphData.nodes)
                .enter()
                .append('g')
                .attr('class', 'node')
                .call(d3.drag()
                    .on('start', dragStarted)
                    .on('drag', dragged)
                    .on('end', dragEnded));

            // Add node shapes based on category
            nodeElements.each(function(d) {{
                const node = d3.select(this);

                // Determine color based on category and cycle status
                let color = d.in_cycle ? 'var(--node-cycle)' : 
                            d.category === 'constructor' ? 'var(--node-constructor)' :
                            d.category === 'property' ? 'var(--node-property)' :
                            d.category === 'async' ? 'var(--node-async)' :
                            d.category === 'private' ? 'var(--node-private)' :
                            d.category === 'method' ? 'var(--node-method)' :
                            'var(--node-function)';

                // Use different shapes for different categories
                if (d.category === 'function') {{
                    node.append('circle')
                        .attr('r', 10)
                        .attr('fill', color)
                        .attr('stroke', 'white')
                        .attr('stroke-width', 1.5);
                }} else {{
                    node.append('rect')
                        .attr('width', 20)
                        .attr('height', 20)
                        .attr('x', -10)
                        .attr('y', -10)
                        .attr('rx', 4)
                        .attr('fill', color)
                        .attr('stroke', 'white')
                        .attr('stroke-width', 1.5);
                }}

                // Add text labels
                node.append('text')
                    .attr('dy', 25)
                    .text(d.name)
                    .attr('fill', 'var(--text-color)')
                    .attr('font-weight', d.in_cycle ? 'bold' : 'normal');
            }});

            // Define arrow markers
            svg.append('defs').selectAll('marker')
                .data(['arrow', 'arrow-cycle'])
                .enter().append('marker')
                .attr('id', d => d)
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 20)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-5L10,0L0,5')
                .attr('fill', d => d === 'arrow-cycle' ? 'var(--link-cycle)' : 'var(--link-normal)');

            // Initialize force simulation
            simulation = d3.forceSimulation(graphData.nodes)
                .force('link', d3.forceLink(graphData.links)
                    .id(d => d.id)
                    .distance(70)
                    .strength(0.6))
                .force('charge', d3.forceManyBody()
                    .strength(-200)
                    .distanceMax(300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collide', d3.forceCollide(30))
                .on('tick', ticked);

            // Add tooltips
            nodeElements
                .on('mouseover', showTooltip)
                .on('mousemove', moveTooltip)
                .on('mouseout', hideTooltip)
                .on('click', highlightRelationships);

            // Hide loading overlay
            loadingOverlay.style.display = 'none';
        }};

        // Update positions on each tick
        function ticked() {{
            linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodeElements
                .attr('transform', d => `translate(${{d.x}}, ${{d.y}})`);
        }}

        // Drag functions
        function dragStarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
            isDragging = true;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragEnded(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            isDragging = false;
        }}

        // Tooltip functions
        function showTooltip(event, d) {{
            // Get category badge
            const badge = `badge-${{d.category}}`;

            // Format full name for display
            let displayName = d.full_name;
            if (displayName.length > 50) {{
                displayName = displayName.substring(0, 47) + '...';
            }}

            // Build HTML content
            tooltip.innerHTML = `
                <h4>${{d.name}}</h4>
                <p><strong>Module:</strong> ${{d.module}}</p>
                ${{d.class ? `<p><strong>Class:</strong> ${{d.class.split('.').pop()}}</p>` : ''}}
                <p><strong>Type:</strong> <span class="badge ${{badge}}">${{d.category}}</span></p>
                ${{d.in_cycle ? '<p><span class="badge badge-cycle">Cyclic Dependency</span></p>' : ''}}
            `;

            // Show and position the tooltip
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY + 10) + 'px';
            tooltip.style.opacity = 1;
        }}

        function moveTooltip(event) {{
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY + 10) + 'px';
        }}

        function hideTooltip() {{
            tooltip.style.opacity = 0;
        }}

        // Highlight related nodes and links
        function highlightRelationships(event, d) {{
            // Skip if dragging
            if (isDragging) return;

            // Reset all nodes and links
            nodeElements.classed('highlighted', false).classed('faded', false);
            linkElements.classed('highlighted', false).classed('faded', false);

            // Get connected node IDs (both incoming and outgoing)
            const connectedNodes = new Set();
            linkElements.each(link => {{
                if (link.source.id === d.id || link.target.id === d.id) {{
                    connectedNodes.add(link.source.id);
                    connectedNodes.add(link.target.id);
                }}
            }});

            // Highlight the selected node and its connections
            nodeElements.classed('highlighted', node => node.id === d.id);
            nodeElements.classed('faded', node => node.id !== d.id && !connectedNodes.has(node.id));

            linkElements.classed('highlighted', link => link.source.id === d.id || link.target.id === d.id);
            linkElements.classed('faded', link => link.source.id !== d.id && link.target.id !== d.id);
        }}

        // Filter nodes and links
        function applyFilters() {{
            const searchTerm = searchInput.value.toLowerCase();
            const moduleValue = moduleFilter.value;
            const typeValue = typeFilter.value;

            nodeElements.each(function(d) {{
                const nameMatch = d.name.toLowerCase().includes(searchTerm) || 
                                 d.full_name.toLowerCase().includes(searchTerm);
                const moduleMatch = !moduleValue || d.module === moduleValue;
                const typeMatch = !typeValue || d.category === typeValue;

                const visible = nameMatch && moduleMatch && typeMatch;
                d3.select(this).style('display', visible ? 'block' : 'none');
            }});

            linkElements.each(function(d) {{
                const sourceVisible = isNodeVisible(d.source);
                const targetVisible = isNodeVisible(d.target);
                d3.select(this).style('display', (sourceVisible && targetVisible) ? 'block' : 'none');
            }});
        }}

        function isNodeVisible(node) {{
            const element = nodeElements.filter(d => d.id === node.id).node();
            return element && getComputedStyle(element).display !== 'none';
        }}

        // Highlight cycle nodes and links
        function highlightCycles() {{
            // Reset all nodes and links
            nodeElements.classed('highlighted', false).classed('faded', true);
            linkElements.classed('highlighted', false).classed('faded', true);

            // Highlight cycle nodes and links
            nodeElements.classed('highlighted', d => d.in_cycle)
                        .classed('faded', d => !d.in_cycle);

            linkElements.classed('highlighted', d => d.is_cycle)
                        .classed('faded', d => !d.is_cycle);
        }}

        // Reset all filters and highlights
        function resetFilters() {{
            searchInput.value = '';
            moduleFilter.value = '';
            typeFilter.value = '';

            nodeElements.style('display', 'block')
                        .classed('highlighted', false)
                        .classed('faded', false);

            linkElements.style('display', 'block')
                        .classed('highlighted', false)
                        .classed('faded', false);
        }}

        // Export SVG
        function exportSVG() {{
            // Create a copy of the SVG with proper sizing
            const svgCopy = svg.node().cloneNode(true);

            // Get the bounding box of all elements
            const bbox = svg.select('g').node().getBBox();

            // Set proper attributes for the copied SVG
            svgCopy.setAttribute('width', bbox.width + 100);
            svgCopy.setAttribute('height', bbox.height + 100);
            svgCopy.setAttribute('viewBox', `${{bbox.x - 50}} ${{bbox.y - 50}} ${{bbox.width + 100}} ${{bbox.height + 100}}`);

            // Add some styling
            const style = document.createElement('style');
            style.textContent = `
                .node circle, .node rect {{ stroke-width: 1.5px; }}
                .node text {{ font-family: Arial; font-size: 10px; }}
                .link {{ stroke-opacity: 0.6; }}
            `;
            svgCopy.insertBefore(style, svgCopy.firstChild);

            // Convert to blob and download
            const svgData = new XMLSerializer().serializeToString(svgCopy);
            const blob = new Blob([svgData], {{ type: 'image/svg+xml' }});
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = '{project_name}_visualization.svg';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}

        // Toggle force simulation
        let simulationPaused = false;
        function toggleForce() {{
            if (simulationPaused) {{
                simulation.alphaTarget(0.1).restart();
                toggleForceBtn.innerHTML = '<i class="fas fa-pause"></i> Pause Layout';
            }} else {{
                simulation.alphaTarget(0).stop();
                toggleForceBtn.innerHTML = '<i class="fas fa-play"></i> Resume Layout';
            }}
            simulationPaused = !simulationPaused;
        }}

        // Zoom controls
        function zoomIn() {{
            svg.transition().duration(300).call(zoom.scaleBy, 1.3);
        }}

        function zoomOut() {{
            svg.transition().duration(300).call(zoom.scaleBy, 0.7);
        }}

        function resetZoom() {{
            svg.transition().duration(300).call(zoom.transform, d3.zoomIdentity);
        }}

        // Event listeners
        document.addEventListener('DOMContentLoaded', initializeVisualization);
        searchInput.addEventListener('input', applyFilters);
        moduleFilter.addEventListener('change', applyFilters);
        typeFilter.addEventListener('change', applyFilters);
        highlightCyclesBtn.addEventListener('click', highlightCycles);
        resetFiltersBtn.addEventListener('click', resetFilters);
        zoomInBtn.addEventListener('click', zoomIn);
        zoomOutBtn.addEventListener('click', zoomOut);
        zoomResetBtn.addEventListener('click', resetZoom);
        exportSvgBtn.addEventListener('click', exportSVG);
        toggleForceBtn.addEventListener('click', toggleForce);

        // Cycle item click handlers
        document.querySelectorAll('#cycles-list .cycle-item').forEach(item => {{
            item.addEventListener('click', () => {{
                const nodes = JSON.parse(item.dataset.nodes);

                // Reset all nodes and links
                nodeElements.classed('highlighted', false).classed('faded', true);
                linkElements.classed('highlighted', false).classed('faded', true);

                // Highlight cycle nodes and their links
                const cycleNodeIds = new Set(nodes);

                nodeElements.classed('highlighted', d => cycleNodeIds.has(d.id))
                            .classed('faded', d => !cycleNodeIds.has(d.id));

                linkElements.classed('highlighted', d => 
                    cycleNodeIds.has(d.source.id) && cycleNodeIds.has(d.target.id))
                            .classed('faded', d => 
                    !(cycleNodeIds.has(d.source.id) && cycleNodeIds.has(d.target.id)));

                // If on mobile, collapse the sidebar
                if (window.innerWidth <= 768) {{
                    sidebar.classList.add('collapsed');
                    sidebarToggle.classList.add('collapsed');
                    sidebarToggle.innerHTML = '<i class="fas fa-chevron-right"></i>';
                    sidebarCollapsed = true;
                }}
            }});
        }});

        // Window resize handler
        window.addEventListener('resize', () => {{
            width = graphContainer.clientWidth;
            height = graphContainer.clientHeight;

            svg.attr('width', width)
               .attr('height', height)
               .attr('viewBox', [0, 0, width, height]);

            simulation.force('center', d3.forceCenter(width / 2, height / 2));
            simulation.alpha(0.3).restart();
        }});
    </script>
</body>
</html>
    """

    return html


def generate_d3_visualization(G: nx.DiGraph, output_path: str, project_name: str):
    """Generate an interactive D3.js visualization."""

    # Extract node data with all necessary attributes
    nodes_data = []
    for node in G.nodes():
        node_type = "method" if G.nodes[node].get("class") else "function"
        module = G.nodes[node].get("module", "")
        class_name = G.nodes[node].get("class", "")
        name = node.split(".")[-1]

        # Get node metadata for display
        is_async = G.nodes[node].get("is_async", False)
        is_property = G.nodes[node].get("is_property", False)
        is_private = name.startswith('_') and not name.startswith('__') and not name.endswith('__')
        is_constructor = name == "__init__" or name == "__new__"

        # Determine node category for styling
        category = "constructor" if is_constructor else \
            "property" if is_property else \
                "async" if is_async else \
                    "private" if is_private else \
                        node_type

        nodes_data.append({
            "id": node,
            "name": name,
            "full_name": node,
            "module": module,
            "class": class_name,
            "category": category,
            "in_cycle": False,  # Will be updated after cycle detection
            "lineno": G.nodes[node].get("lineno", 0),
            "file_path": G.nodes[node].get("path", "")
        })

    # Extract edge data
    links_data = []
    for source, target, data in G.edges(data=True):
        links_data.append({
            "source": source,
            "target": target,
            "is_cycle": data.get("is_cycle", False),
            "lineno": data.get("lineno", 0)
        })

    # Detect cycles
    try:
        cycles = list(nx.simple_cycles(G))

        # Mark nodes that are part of cycles
        cycle_nodes = set()
        for cycle in cycles:
            for node in cycle:
                cycle_nodes.add(node)

        # Update node data with cycle information
        for node_data in nodes_data:
            if node_data["id"] in cycle_nodes:
                node_data["in_cycle"] = True

        # Create cycle metadata for display
        cycles_info = []
        for i, cycle in enumerate(cycles):
            cycles_info.append({
                "id": i,
                "nodes": cycle,
                "length": len(cycle),
                "description": " → ".join([n.split(".")[-1] for n in cycle + [cycle[0]]])
            })

    except Exception as e:
        logger.warning(f"Could not detect cycles: {str(e)}")
        cycles_info = []

    # Prepare node and link data for D3
    graph_data = {
        "nodes": nodes_data,
        "links": links_data,
        "cycles": cycles_info,
        "project_info": {
            "name": project_name,
            "total_nodes": len(nodes_data),
            "total_links": len(links_data),
            "modules": list(set(n["module"] for n in nodes_data)),
            "total_cycles": len(cycles_info)
        }
    }

    # Generate HTML with embedded D3.js visualization
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(create_d3_html_template(graph_data, project_name))

    logger.info(f"D3.js visualization saved to {output_path}")
    return True




def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Generate method call diagrams for Python projects',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('path', help='Path to Python project or file')
    parser.add_argument('--output', '-o', help='Output file path (default: stdout)')
    parser.add_argument(
        '--format', '-f',
        choices=['html', 'mermaid', 'svg', 'png'],
        default='html',
        help='Output format: html (interactive D3.js), mermaid (simple diagram), svg/png (static image)'
    )
    parser.add_argument('--modules', '-m', nargs='+', help='Filter by module names')
    parser.add_argument('--depth', '-d', type=int, help='Maximum call depth from entry point')
    parser.add_argument('--entry', '-e', help='Entry point function (format: module.function)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--exclude', '-x', nargs='+', help='Exclude modules matching these patterns')
    parser.add_argument('--max-nodes', type=int, default=150, help='Maximum number of nodes to include in the diagram')
    parser.add_argument('--project-name', '-p', help='Project name to use in diagram title')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Normalize project path to absolute path
    project_path = os.path.abspath(args.path)

    # Get project name from path or argument
    project_name = args.project_name or os.path.basename(project_path)

    # Only find Python files that are within the project directory
    py_files = find_project_python_files(project_path)
    if not py_files:
        logger.error(f"No Python files found in {args.path}")
        sys.exit(1)

    # Get project root (either the provided path or common parent directory)
    project_root = project_path if os.path.isdir(project_path) else os.path.dirname(project_path)

    # Analyze the project to build a comprehensive understanding
    logger.info("Analyzing project structure and dependencies...")
    module_analyzers, all_calls = analyze_project(py_files, project_root)

    # Build the call graph with improved resolution
    logger.info("Building function call graph...")
    G = build_call_graph(module_analyzers, all_calls)
    logger.info(f"Built graph with {len(G.nodes())} functions and {len(G.edges())} calls")

    # Apply filters
    if args.modules:
        logger.info(f"Filtering to include only modules: {', '.join(args.modules)}")
        G = filter_by_modules(G, args.modules)
        logger.info(f"After module filtering: {len(G.nodes())} functions and {len(G.edges())} calls")

    if args.exclude:
        logger.info(f"Excluding modules: {', '.join(args.exclude)}")
        nodes_to_remove = []
        for node in G.nodes():
            module = G.nodes[node].get('module', '')
            if any(module.startswith(excluded) for excluded in args.exclude):
                nodes_to_remove.append(node)
        G.remove_nodes_from(nodes_to_remove)
        logger.info(f"After exclusion: {len(G.nodes())} functions and {len(G.edges())} calls")

    if args.entry and args.depth:
        logger.info(f"Filtering to depth {args.depth} from entry point {args.entry}")
        G = filter_by_depth(G, args.entry, args.depth)
        logger.info(f"After depth filtering: {len(G.nodes())} functions and {len(G.edges())} calls")

    # Limit the number of nodes if needed
    if len(G.nodes()) > args.max_nodes:
        logger.warning(f"Graph has {len(G.nodes())} nodes, which exceeds the limit of {args.max_nodes}")
        logger.warning("Removing least connected nodes to reduce graph size")

        # Sort nodes by degree (number of connections)
        node_degrees = sorted(G.degree(), key=lambda x: x[1])
        nodes_to_remove = [node for node, degree in node_degrees[:len(G.nodes()) - args.max_nodes]]
        G.remove_nodes_from(nodes_to_remove)
        logger.info(f"After limiting nodes: {len(G.nodes())} functions and {len(G.edges())} calls")

    # Generate output
    if len(G.nodes()) == 0:
        logger.warning("No functions to visualize after applying filters")
        sys.exit(0)

    # Set default output path if not provided
    if not args.output:
        args.output = f"{project_name}_visualization.{args.format}"

    # Ensure the directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate the appropriate visualization based on format
    if args.format == 'html':
        logger.info("Generating interactive D3.js visualization...")
        generate_d3_visualization(G, args.output, project_name)
    elif args.format == 'mermaid':
        logger.info("Generating Mermaid diagram...")
        mermaid_code = generate_styled_mermaid(G)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        # Also generate HTML version for better viewing
        html_path = f"{os.path.splitext(args.output)[0]}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(create_interactive_html(mermaid_code, project_name))
        logger.info(f"Mermaid diagram saved to {args.output}")
        logger.info(f"Interactive HTML version saved to {html_path}")
    elif args.format in ['svg', 'png']:
        try:
            import graphviz
            logger.info(f"Generating {args.format.upper()} using Graphviz...")

            # Create graphviz graph
            dot = graphviz.Digraph(
                comment=f'{project_name} Code Structure',
                engine='dot',
                format=args.format,
                graph_attr={
                    'rankdir': 'LR',
                    'bgcolor': 'transparent',
                    'fontname': 'Arial',
                    'nodesep': '0.8',
                    'ranksep': '1.0'
                }
            )

            # Add nodes with styling
            for node in G.nodes():
                node_name = node.split('.')[-1]
                node_type = "method" if G.nodes[node].get("class") else "function"

                # Determine fill color based on type and properties
                if node_name == "__init__" or node_name == "__new__":
                    fillcolor = "#E53935"  # Constructor red
                elif G.nodes[node].get("is_property", False):
                    fillcolor = "#FF6D00"  # Property orange
                elif G.nodes[node].get("is_async", False):
                    fillcolor = "#AA00FF"  # Async purple
                elif node_name.startswith('_') and not node_name.startswith('__') and not node_name.endswith('__'):
                    fillcolor = "#757575"  # Private gray
                elif node_type == "method":
                    fillcolor = "#2962FF"  # Method blue
                else:
                    fillcolor = "#00C853"  # Function green

                # Add the node with styling
                dot.node(
                    node,
                    label=node_name,
                    shape='box' if node_type == 'method' else 'ellipse',
                    style='filled',
                    fillcolor=fillcolor,
                    fontcolor='white',
                    fontname='Arial',
                    fontsize='12'
                )

            # Add edges with styling
            for source, target, data in G.edges(data=True):
                is_cycle = data.get('is_cycle', False)

                if is_cycle:
                    dot.edge(source, target, color='#F44336', style='dashed', penwidth='1.5')
                else:
                    dot.edge(source, target, color='#616161', penwidth='1.0')

            # Render and save
            dot.render(args.output, cleanup=True)
            logger.info(f"{args.format.upper()} saved to {args.output}.{args.format}")

        except (ImportError, Exception) as e:
            logger.error(f"Failed to generate {args.format}: {str(e)}")
            logger.warning(f"Falling back to D3.js HTML visualization")
            generate_d3_visualization(G, f"{os.path.splitext(args.output)[0]}.html", project_name)
    else:
        # This shouldn't happen due to argument choices, but just in case
        logger.error(f"Unsupported output format: {args.format}")

if __name__ == "__main__":
    main()
