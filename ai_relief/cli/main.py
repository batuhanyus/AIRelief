import typer
from pathlib import Path
from ai_relief.config.settings import settings

app = typer.Typer(help="AI Relief CLI - Convert 2D photos to 3D bas-reliefs")

@app.command()
def process(
    input_image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, help="Path to the input 2D image"),
    output_mesh: Path = typer.Argument(..., file_okay=True, dir_okay=False, resolve_path=True, help="Path to save the output 3D mesh (e.g. output.stl)"),
    max_depth: float = typer.Option(5.0, help="Maximum thickness of the relief in mm"),
    base_thickness: float = typer.Option(2.0, help="Thickness of the solid base in mm"),
    device: str = typer.Option("auto", help="Device to use for inference (auto, cuda, mps, cpu)")
):
    """
    Process a single image into a 3D bas-relief mesh.
    """
    settings.relief_max_depth_mm = max_depth
    settings.base_thickness_mm = base_thickness
    settings.device = device
    
    typer.echo(f"Starting AI Relief Processing...")
    typer.echo(f"Input: {input_image}")
    typer.echo(f"Output: {output_mesh}")
    typer.echo(f"Settings: max_depth={max_depth}mm, base={base_thickness}mm, device={device}")
    
    # TODO: Connect the actual AI pipeline here once implemented
    typer.echo("Pipeline is under construction.")
    
if __name__ == "__main__":
    app()
