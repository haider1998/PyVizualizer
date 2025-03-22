# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="py-code-visualizer",
    version="1.0.0",
    author="Syed Mohd Haider Rizvi",
    author_email="smhrizvi281@gmail.com",
    description="Architectural intelligence for Python codebases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haider1998/pyvisualizer",
    project_urls={
        "Bug Tracker": "https://github.com/haider1998/pyvisualizer/issues",
        "Documentation": "https://github.com/haider1998/pyvisualizer/wiki",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Documentation",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.5",
        "astroid>=2.5.0",
        "lru-dict>=1.1.7",
    ],
    entry_points={
        "console_scripts": [
            "pyvisualizer=pyvisualizer.main:main",
        ],
    },
)
