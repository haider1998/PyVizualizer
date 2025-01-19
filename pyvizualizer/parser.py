import ast
import os

class PyVizualizerParser:
    """
    Parses Python source code to extract class and method information,
    identifying inheritance relationships.
    """

    def __init__(self, root_dir, excludes=None):
        self.root_dir = root_dir
        self.excludes = excludes or []  # List of files or directories to exclude
        self.class_data = {}

    def parse(self):
        """
        Parses the Python files in the specified directory.
        """
        for root, _, files in os.walk(self.root_dir):
            if any(exclude in root for exclude in self.excludes):
                continue
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    self.parse_file(filepath)

    def parse_file(self, filepath):
        """
        Parses a single Python file to extract class and method details.
        """
        try:
            with open(filepath, "r") as f:
                source = f.read()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        self.parse_class(node, filepath)
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}")

    def parse_class(self, node, filepath):
        """
        Parses a class definition to record its name, methods, and inheritance.
        """
        class_name = node.name
        methods = [
            n.name
            for n in ast.walk(node)
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
        ]
        # Handling simple inheritance, getting base class names
        bases = [base.id for base in node.bases if isinstance(base, ast.Name)]

        # Add or update class data
        if class_name not in self.class_data:
            self.class_data[class_name] = {"methods": methods, "bases": bases, "filepath": filepath}
        else:
            # Update existing class data
            existing_data = self.class_data[class_name]
            existing_data["methods"].extend(
                method for method in methods if method not in existing_data["methods"]
            )
            existing_data["bases"].extend(
                base for base in bases if base not in existing_data["bases"]
            )
            if existing_data["filepath"] != filepath:
                print(
                    f"Warning: Class {class_name} is defined in multiple files: {existing_data['filepath']}, {filepath}"
                )
            existing_data["filepath"] = filepath  # Update with the latest file path if needed

    def get_class_data(self):
        """
        Returns the collected class data.
        """
        return self.class_data