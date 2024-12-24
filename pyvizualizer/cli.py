# pyvizualizer/cli.py

import argparse
import logging
import subprocess
import sys
from .analyzer import Analyzer
from .mermaid_generator import MermaidGenerator
from .utils import setup_logging

def run_script():
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pyvizualizer.cli', 'examples/test_project'],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="PyVizualizer: Visualize Python project workflows.")
    parser.add_argument("project_path", help="Path to the Python project to analyze")
    parser.add_argument("--output", "-o", help="Path to output the Mermaid file", default="diagram.mmd")
    parser.add_argument("--log-level", help="Logging level", default="INFO")
    args = parser.parse_args()

    logger = setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        analyzer = Analyzer(args.project_path)
        graph = analyzer.analyze()

        mermaid_generator = MermaidGenerator(graph)
        mermaid_code = mermaid_generator.generate()

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        logger.info(f"Mermaid diagram written to {args.output}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_script()