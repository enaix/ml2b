import typer
from src.runners import DockerRunner
from pathlib import Path

app = typer.Typer(invoke_without_command=True)
from loguru import logger
from typing import Annotated
from environments import build_agent, BuildArgs

DEFAULT_CONFIG_PATH = Path(__file__) / "environments" / "container_config.yaml"
DEFAULT_DATA_PATH = Path(__file__) / "data"


@app.command()
def run_bench(
    image_name: Annotated[str, typer.Option(help="Agent image name")],
    competition_set: Annotated[
        str, typer.Option(help="Path to file with competition script")
    ],
    num_workers: Annotated[
        int, typer.Option(help="Number of workers to parallel run")
    ] = 1,
    data_dir: Annotated[
        str, typer.Option(help="Path to directory with data for bench")
    ] = DEFAULT_DATA_PATH,
    container_config: Annotated[str, typer.Option(help="")] = DEFAULT_CONFIG_PATH,
):
    """
    Run banch for a specified agent
    """
    


@app.command()
def build_agent_runtime(
    image_name: Annotated[
        str, typer.Option(help="Target container name")
    ] = "runtime-env",
    dockerfile_path: Annotated[
        str, typer.Option(help="Build image for agent")
    ] = "environments/runtime/Dockerfile",
    platform: Annotated[
        str, typer.Option(help="Target container arch")
    ] = "linux/amd64",
) -> None:
    """
    Build base runtime and agent images
    """
    build_args = BuildArgs(
        tag=image_name, dockerfile=dockerfile_path, platform=platform
    )
    build_agent(build_args)


@app.callback(invoke_without_command=True)
def help(ctx: typer.Context):
    """
    For default help without arguments
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":
    app()
