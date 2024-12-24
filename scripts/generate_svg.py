# scripts/generate_svg.py

import os
import subprocess
import sys
import logging

def generate_svg(mermaid_file_path, output_path):
    """Generates an SVG image from a Mermaid file."""
    logger = logging.getLogger('pyvizualizer')
    logger.info(f"Generating SVG from Mermaid file: {mermaid_file_path}")
    try:
        subprocess.run(
            ["mmdc", "-i", mermaid_file_path, "-o", output_path],
            check=True
        )
        logger.info(f"SVG generated at: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate SVG: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SVG from Mermaid file.")
    parser.add_argument("mermaid_file", help="Path to the input Mermaid (.mmd) file")
    parser.add_argument("output_file", help="Path to the output SVG file")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    generate_svg(args.mermaid_file, args.output_file)
