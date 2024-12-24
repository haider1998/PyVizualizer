# pyvizualizer/utils.py

import os
import logging
import sys

def setup_logging(log_level=logging.INFO):
    """Sets up the logging configuration."""
    logger = logging.getLogger('pyvizualizer')
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def get_python_files(directory):
    """Recursively gets all Python files in a directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files
