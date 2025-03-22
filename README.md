<div align="center">

![PyVisualizer Logo](docs/images/PyVizualizer_Logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/pyvisualizer.svg)](https://pypi.org/project/pyvisualizer/)
[![Downloads](https://static.pepy.tech/personalized-badge/pyvisualizer?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/pyvisualizer)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Architectural intelligence for Python codebases. Transform complex systems into stunning interactive diagrams.**

[Features](#-key-features) • 
[Installation](#-installation) • 
[Examples](#-visualizations) • 
[Usage](#-quick-start) • 
[Documentation](#-documentation) • 
[Contributing](#-contributing)

</div>

## 🔍 What is PyVisualizer?

PyVisualizer is a powerful tool that transforms complex Python codebases into intuitive visual diagrams. Created for developers who need to understand large systems quickly, it illuminates the hidden architecture of your projects with beautiful, interactive visualizations.

Instead of spending hours tracing through imports and function calls manually, PyVisualizer automatically maps relationships between modules, classes, and methods, making them instantly comprehensible.

> *"PyVisualizer transformed how we onboard engineers to our 250K+ LOC Python codebase. What took days now takes hours."* — Senior Engineering Manager at a Fortune 500 company

## 🌟 Key Features

- **Interactive Architecture Maps** — Visualize inheritance chains, method calls, and module dependencies with powerful filtering and search capabilities
- **Smart Code Analysis** — Advanced parsing of Python's AST to detect relationships without executing code
- **Framework-Aware** — Special detection for Flask/Django routes, FastAPI endpoints, and modern framework patterns
- **Full Python Support** — Properly handles decorators, async functions, properties, type hints, and other advanced Python features
- **Multi-format Export** — Generate interactive HTML, publication-quality SVG, or PNG diagrams for documentation
- **Performance Optimized** — Efficiently analyzes large codebases with 500K+ lines of code through parallel processing
- **Beautiful UI** — Dark/light mode, zoom controls, interactive filtering, and search functionality in generated diagrams

## 📊 Visualizations

<div align="center">
<img src="pyvisualizer/docs/images/FatigueFinder_methods.svg" alt="PyVisualizer Example" width="85%">
<br>
<em>Interactive diagram of a ML application's architecture</em>
</div>

## 💻 Installation

```bash
# Via pip (recommended)
pip install pyvisualizer

# From source
git clone https://github.com/haider1998/PyVisualizer.git
cd PyVisualizer
pip install -e .
```

## 🚀 Quick Start

### Visualize an entire project
```bash
pyvisualizer /path/to/your/project -o architecture.html
```

### Trace specific execution flows
```bash
# Visualize execution flow from an entry point, limited to 3 levels deep
pyvisualizer /path/to/your/project -e app.main.start_server -d 3 -o execution_flow.svg 
```

### Focus on specific modules
```bash
# Generate diagram focused only on core components
pyvisualizer /path/to/your/project -m core.services api.routes -o core_components.html
```

## 🛠️ Advanced Usage

### Command Line Options

```
pyvisualizer [OPTIONS] PROJECT_PATH
```

| Option | Description |
|--------|-------------|
| `path` | Path to Python project or file |
| `-o, --output` | Output file path |
| `-f, --format` | Format: `mermaid`, `svg`, `png`, `html` (default: `html`) |
| `-m, --modules` | Include only specified modules |
| `-x, --exclude` | Exclude specified modules |
| `-e, --entry` | Entry point function (format: module.function) |
| `-d, --depth` | Maximum call depth from entry point (default: 3) |
| `-v, --verbose` | Enable detailed logging |
| `--max-nodes` | Maximum nodes in diagram (default: 150) |

### CI/CD Integration

Keep architecture diagrams current by integrating with your CI/CD pipeline:

```yaml
# GitHub Actions example
steps:
  - name: Generate Architecture Diagram
    run: |
      pip install pyvisualizer
      pyvisualizer . -o docs/architecture.svg
      git config user.name github-actions
      git config user.email github-actions@github.com
      git add docs/architecture.svg
      git commit -m "Update architecture diagram" || echo "No changes"
      git push
```

## 📘 Documentation

Comprehensive documentation is available at our [GitHub Wiki](https://github.com/haider1998/PyVisualizer/wiki):

- [User Guide](https://github.com/haider1998/PyVisualizer/wiki/User-Guide) - Detailed instructions on using PyVisualizer
- [API Reference](https://github.com/haider1998/PyVisualizer/wiki/API-Reference) - Complete reference for integrating PyVisualizer into your own tools
- [Advanced Techniques](https://github.com/haider1998/PyVisualizer/wiki/Advanced-Techniques) - Tips and tricks for power users
- [Customization Guide](https://github.com/haider1998/PyVisualizer/wiki/Customization-Guide) - How to customize the visualization output

## 🧩 How It Works

PyVisualizer leverages Python's Abstract Syntax Tree (AST) to analyze your code without executing it:

1. **Project Scanning** - Discovers Python files while respecting common exclusion patterns
2. **AST Analysis** - Parses code to extract classes, methods, and their relationships
3. **Dependency Resolution** - Builds a complete map of imports and calls between components  
4. **Graph Construction** - Creates a directed graph representing your code's architecture
5. **Visual Rendering** - Transforms the graph into beautiful, interactive visualizations

## 🚀 Use Cases

### For Engineering Teams
- **New Developer Onboarding** - Provide an instant overview of system architecture
- **Architecture Documentation** - Maintain living documentation that updates with your code
- **Code Reviews** - Visualize architectural impacts of proposed changes

### For Architects & Tech Leads
- **Refactoring Planning** - Identify highly coupled components and architectural boundaries
- **Technical Presentations** - Create compelling visuals for architecture discussions
- **Technical Debt Management** - Spot unexpected dependencies and architecture violations

## 🤝 Contributing

Contributions are welcome from developers of all skill levels! See our [contributing guidelines](CONTRIBUTING.md) for how to get started.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/haider1998/PyVisualizer.git
cd PyVisualizer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## 👨‍💻 About the Author

**Syed Mohd Haider Rizvi** is a software architect specializing in Python systems analysis and visualization tools.

<div align="center">
  <a href="mailto:smhrizvi281@gmail.com"><img src="https://img.shields.io/badge/Email-smhrizvi281%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
  <a href="https://github.com/haider1998"><img src="https://img.shields.io/badge/GitHub-haider1998-181717?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://www.linkedin.com/in/s-m-h-rizvi-0a40441ab/"><img src="https://img.shields.io/badge/LinkedIn-S.M.H._Rizvi-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>
</div>

## 📃 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>
    <i>If PyVisualizer helps your team, please consider giving it a ⭐️ on GitHub!</i>
  </p>
  <a href="https://github.com/haider1998/PyVisualizer">
    <img src="https://img.shields.io/github/stars/haider1998/PyVisualizer?style=social" alt="GitHub stars">
  </a>
</div>