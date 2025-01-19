# pyvizualizer/graph.py

from typing import Dict, List, Set, Tuple
import logging

logger = logging.getLogger('pyvizualizer')

class Node:
    """Represents a node in the code graph."""

    def __init__(self, name: str, node_type: str):
        self.name = name
        self.type = node_type  # 'class' or 'function'
        self.calls = set()     # Set of node names that this node calls

    def __repr__(self):
        return f"Node(name={self.name}, type={self.type})"

    def add_call(self, callee_name: str):
        print(f"Node {self.name} calls {callee_name}")
        self.calls.add(callee_name)


class CodeGraph:
    """Represents the entire code graph."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Tuple[str, str]] = []

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
                    print(f"Created class node: {class_node_name}")
                    for method in item['methods']:
                        method_node_name = f"{class_node_name}.{method['name']}"
                        method_node = Node(name=method_node_name, node_type='method')
                        self.nodes[method_node_name] = method_node
                        print(f"Created method node: {method_node_name}")
                elif item['type'] == 'function':
                    function_node_name = f"{module}.{item['name']}"
                    function_node = Node(name=function_node_name, node_type='function')
                    self.nodes[function_node_name] = function_node
                    print(f"Created function node: {function_node_name}")

        # Then, add edges
        for module, defs in definitions.items():
            for item in defs:
                if item['type'] == 'class':
                    class_node_name = f"{module}.{item['name']}"
                    for method in item['methods']:
                        method_node_name = f"{class_node_name}.{method['name']}"
                        for called_name in method['calls']:
                            if called_name in self.nodes:
                                self.nodes[method_node_name].add_call(called_name)
                                self.add_edge(method_node_name, called_name)
                elif item['type'] == 'function':
                    function_node_name = f"{module}.{item['name']}"
                    for called_name in item['calls']:
                        if called_name in self.nodes:
                            self.nodes[function_node_name].add_call(called_name)
                            self.add_edge(function_node_name, called_name)

    def add_edge(self, from_node: str, to_node: str):
        """Adds an edge to the graph."""
        print(f"Adding edge from {from_node} to {to_node}")
        self.edges.append((from_node, to_node))

    def get_edges(self) -> List[Tuple[str, str]]:
        """Returns the list of edges."""
        return self.edges