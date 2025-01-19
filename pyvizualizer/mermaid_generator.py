# pyvizualizer/mermaid_generator.py

import logging
from .graph import CodeGraph
from .exceptions import MermaidGenerationError

logger = logging.getLogger('pyvizualizer')

class MermaidGenerator:
    """Generates Mermaid diagrams from the code graph."""

    def __init__(self, graph: CodeGraph):
        self.graph = graph

    def generate(self):
        """Generates Mermaid code representing the code graph."""
        try:
            logger.info("Generating Mermaid code")
            lines = ["flowchart TD"]
            lines.append("%% Nodes")
            for node_name, node in self.graph.nodes.items():
                node_id = self._sanitize_node_id(node_name)
                lines.append(f'    {node_id}("{self._format_node_label(node_name)}")')
            lines.append("\n%% Edge connections between nodes")
            for node_name, node in self.graph.nodes.items():
                node_id = self._sanitize_node_id(node_name)
                for callee_name in node.calls:
                    callee_id = self._sanitize_node_id(callee_name)
                    lines.append(f'    {node_id} --> {callee_id}')
            lines.append("\n%% Individual node styling. Try the visual editor toolbar for easier styling!")
            # Add custom styles if needed
            # lines.append('    style E color:#FFFFFF, fill:#AA00FF, stroke:#AA00FF')
            mermaid_code = '\n'.join(lines)
            return mermaid_code
        except Exception as e:
            logger.error(f"Failed to generate Mermaid code: {e}")
            raise MermaidGenerationError("An error occurred during Mermaid code generation") from e

    def _sanitize_node_id(self, name):
        """Sanitizes a node name to be used as a Mermaid node identifier."""
        return name.replace('.', '_').replace('<', '').replace('>', '')

    def _format_node_label(self, name):
        """Formats the node label for better readability."""
        return name.split('/')[-1].replace('_', ' ').replace('.py', '').replace('.', ' ')