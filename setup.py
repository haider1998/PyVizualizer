# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="pyvizualizer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Visualize Python project workflows using Mermaid diagrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyvizualizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        # List your project's dependencies here.
    ],
    entry_points={
        'console_scripts': [
            'pyvizualizer=pyvizualizer.cli:main',
        ],
    },
)
