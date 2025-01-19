import click
from pyvizualizer.parser import PyVizualizerParser
from pyvizualizer.mermaid_generator import MermaidGenerator
import os

@click.command()
@click.argument(
    "root_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=".",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False),
    default="class_diagram.mmd",
    help="Output file name for the Mermaid diagram.",
)
@click.option(
    "--layout",
    "-l",
    type=click.Choice(["TB", "LR"], case_sensitive=False),
    default="TB",
    help="Layout direction of the diagram (TB for top-to-bottom, LR for left-to-right).",
)
@click.option(
    "--excludes",
    "-e",
    multiple=True,
    default=[],
    help="Files or directories to exclude from parsing.",
)
def cli(root_dir, output, layout, excludes):
    """
    PyVizualizer: A tool to visualize the architecture of Python projects.
    Parses Python code and generates class diagrams using Mermaid syntax.
    """
    click.echo(f"Parsing directory: {root_dir}")

    parser = PyVizualizerParser(root_dir, excludes=excludes)
    parser.parse()
    class_data = parser.get_class_data()

    generator = MermaidGenerator(class_data, layout_direction=layout)
    diagram = generator.generate_diagram()

    # Determine file extension and generate accordingly
    output_ext = os.path.splitext(output)[1].lower()

    if output_ext == ".mmd":
        # Save as .mmd file
        with open(output, "w") as f:
            f.write(diagram)
        click.echo(f"Mermaid diagram saved to {output}")
    elif output_ext == ".html":
        # Generate HTML with embedded Mermaid diagram
        html_content = generate_html(diagram)
        with open(output, "w") as f:
            f.write(html_content)
        click.echo(f"HTML file saved to {output}")
    else:
        click.echo(f"Unsupported output format: {output_ext}")

    # Optional: Output to console
    if click.confirm("Do you want to print the diagram to the console?", default=False):
        click.echo(diagram)

def generate_html(diagram_content):
    """
    Generates an HTML file with an embedded Mermaid diagram.

    Args:
        diagram_content: The Mermaid diagram code as a string.

    Returns:
        A string containing the HTML content.
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mermaid Diagram</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{startOnLoad: true}});
        </script>
    </head>
    <body>
        <pre class="mermaid">
            {diagram_content}
        </pre>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    cli()