# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="pyvizualizer",
    version="0.1.0",
    author="Syed Mohd Haider Rizvi",
    author_email="smhrizvi281@gmail.com",
    description="Visualize Python project workflows using Mermaid diagrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haider1998/PyVizualizer",  # Link to your project's homepage
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        # List your project's dependencies here.
        # For example: 'numpy', 'requests', etc.
    ],
    entry_points={
        'console_scripts': [
            'pyvizualizer=pyvizualizer.cli:main',
        ],
    },
)
