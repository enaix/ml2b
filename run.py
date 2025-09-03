"""
Module for simplifying benchmark setup and execution.
"""
import typer
from pathlib import Path
from src.main import run_benchmark
from loguru import logger
from typing import Annotated, Literal
from environments import build_agent, BuildArgs
from src.runners import RunnerSpec
from src.utils.setup_logger import setup_logger

app = typer.Typer(invoke_without_command=True)

DEFAULT_CONFIG_PATH = Path(__file__) / "environments" / "container_config.yaml"
DEFAULT_DATA_PATH = Path(__file__) / "data"

@app.command()
def run_bench(
    image_name: Annotated[str, typer.Option(help="Agent image name")],
    competition_set: Annotated[
        str, typer.Option(help="Path to file with competition set")
    ],
    num_workers: Annotated[
        int, typer.Option(help="Number of workers to parallel run")
    ] = 1,
    data_dir: Annotated[
        str, typer.Option(help="Path to directory with data for bench")
    ] = DEFAULT_DATA_PATH,
    container_config: Annotated[str, typer.Option(help="")] = DEFAULT_CONFIG_PATH,
    log_level: Literal["TRACE", "DEBUG", "INFO"] = "INFO",
    log_folder: Annotated[
        str | None, typer.Option(help="Path to bench log folder")
    ] = None
 ) -> None:
    """
    Run banch for a specified agent
    """
    runner_spec = RunnerSpec(
        image_name=image_name,
        num_workers=num_workers,
        competition_set=Path(competition_set).resolve(),
        data_dir=Path(data_dir).resolve(),
        container_config=Path(container_config).resolve()
    )
    setup_logger(log_level, log_folder)
    logger.info(f"[green]Run benchmark with runner spec: {runner_spec}[/green]")
    run_benchmark(runner_spec)


@app.command()
def build_agent_runtime(
    image_name: Annotated[
        str, typer.Option(help="Target container name same as agent name")
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
def help(ctx: typer.Context) -> None:
    """
    For default help without arguments
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":
    app()
