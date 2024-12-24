# pyvizualizer/graph.py

from typing import Dict, List, Set
import logging

logger = logging.getLogger('pyvizualizer')

class Node:
    """Represents a node in the code graph."""

    def __init__(self, name: str, node_type: str):
        self.name = name
        self.type = node_type  # 'class' or 'function'
        self.calls = set()     # Set of node names that this node calls

    def add_call(self, callee_name: str):
        logger.debug(f"Node {self.name} calls {callee_name}")
        self.calls.add(callee_name)


class CodeGraph:
    """Represents the entire code graph."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def build_graph(self, definitions: Dict[str, Dict]):
        """Builds the graph from parsed definitions."""
        logger.info("Building code graph")
        # First, create all nodes
        for module, defs in definitions.items():
            for item in defs:
                if item['type'] == 'class':
                    class_node_name = f"{module}.{item['name']}"
                    class_node = Node(name=class_node_name, node_type='class')
                    self.nodes[class_node_name] = class_node
                    logger.debug(f"Created class node: {class_node_name}")
                    for method in item['methods']:
                        method_node_name = f"{class_node_name}.{method['name']}"
                        method_node = Node(name=method_node_name, node_type='method')
                        self.nodes[method_node_name] = method_node
                        logger.debug(f"Created method node: {method_node_name}")
                        # Add calls
                        for called_name in method['calls']:
                            # Only add edges to nodes within the project
                            if called_name in self.nodes:
                                method_node.add_call(called_name)
                elif item['type'] == 'function':
                    function_node_name = f"{module}.{item['name']}"
                    function_node = Node(name=function_node_name, node_type='function')
                    self.nodes[function_node_name] = function_node
                    logger.debug(f"Created function node: {function_node_name}")
                    # Add calls
                    for called_name in item['calls']:
                        if called_name in self.nodes:
                            function_node.add_call(called_name)