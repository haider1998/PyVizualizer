class MermaidGenerator:
    """
    Generates Mermaid class diagrams from class data.
    """

    def __init__(self, class_data, layout_direction="TB"):
        self.class_data = class_data
        self.layout_direction = layout_direction

    def generate_diagram(self):
        """
        Generates a Mermaid class diagram from the parsed class data.
        """
        diagram = [f"classDiagram\n  direction {self.layout_direction}"]
        for class_name, details in self.class_data.items():
            diagram.append(f"  class {class_name} {{")
            for method in details["methods"]:
                diagram.append(f"    + {method}()")
            diagram.append("  }")
            for base in details["bases"]:
                diagram.append(f"  {base} <|-- {class_name}")
        return "\n".join(diagram)