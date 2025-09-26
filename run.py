"""
Module for simplifying benchmark setup and execution.
"""
from pathlib import Path
from src.main import run_benchmark
from loguru import logger
from environments import build_image, BuildArgs
from src.runners import RunnerSpec
from src.utils.setup_logger import setup_logger
import click
import docker
import time
from competitions import load_data


@click.group()
def cli():
    pass

DEFAULT_TAG = "runtime-env"
BASE_PATH = Path(__file__).parent
DEFAULT_RUNTIME_PATH = BASE_PATH / "environments" / "runtime"
DEFAULT_CONFIG_PATH = BASE_PATH / "environments" / "container_config.yaml"
DEFAULT_DATA_PATH = BASE_PATH / "competitions" / "data"
DEFAULT_LOGS_PATH = BASE_PATH / "logs"
DEFAULT_COMPETITIONS_PATH = BASE_PATH / "competitions"

@cli.command()
@click.option(
    "-i", 
    "--image-name", 
    type=click.STRING, 
    default=DEFAULT_TAG,
    help="Target container name, usualy same as agent name"
    )
@click.option(
    '--agent-dir',
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
    default=DEFAULT_RUNTIME_PATH,
    help="Agent directory"
)
@click.option(
    '--platform',
    type=click.STRING,
    default="linux/amd64",
    help="Target container arch"
)
def build_runtime(image_name: str, agent_dir: Path, platform: str) -> None:
    """
    Build base runtime and agent images
    """
    client = docker.from_env()
    build_args = BuildArgs(
        tag=image_name, 
        dockerfile="Dockerfile", 
        platform=platform,
        path=str(agent_dir)
    )
    build_image(BuildArgs(
        tag=DEFAULT_TAG,
        dockerfile=DEFAULT_RUNTIME_PATH / "Dockerfile",
        platform=platform
    ), client)
    if DEFAULT_TAG != image_name:
        build_image(build_args, client)


@cli.command()
@click.option(
    "-i", 
    "--image-name", 
    type=click.STRING,
    help="Agent image name" 
)
@click.option(
    "-w",
    "--workers", 
    type=click.IntRange(1, 100),
    help="Num parralel workers to run" 
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
    default=DEFAULT_DATA_PATH,
    help="Agent directory"
)
@click.option(
    "--runtime-config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    default=DEFAULT_RUNTIME_PATH / "runtime_config.json",
    help="Path container params config",
)
@click.option(
    "--log-level",
    type=click.Choice(["TRACE", "DEBUG", "INFO", "WARNING"], case_sensitive=False),
    default="INFO",
    help="Logger level"
)
@click.option(
    "--logs-dir",
    type=click.Path(exists=False, file_okay=False, readable=True, path_type=Path),
    default=DEFAULT_LOGS_PATH,
    help="Logs directory"
)
@click.option(
    "--competitions",
    type=click.Path(exists=True, dir_okay=True, readable=True, path_type=Path),
    default=DEFAULT_COMPETITIONS_PATH,
    help="Competitions path"
)
@click.option(
    "--folds",
    type=click.INT,
    default=1,
    help="Competitions split to run"
)
@click.option(
    "--seed",
    type=click.INT,
    default=None,
    help="Randomness seed"
)
@click.option(
    "--code-variant",
    type=click.Choice(["extended", "short"]),
    default="extended",
    help="Extended or short code variant bechmark"
)
@click.option(
    "--agent-dir",
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
    help="Agent directory"
)
@click.option(
    "--network",
    type=click.STRING,
    default=None,
    help="Network name for agent container"
)
@click.option(
    "--args-variant",
    type=click.Choice(["extended", "short"]),
    default="short",
    help="Extended or short code variant bechmark"
)
def bench(image_name: str, workers: int, data_dir: Path, 
          runtime_config: Path, log_level: str, logs_dir: Path, 
          competitions: Path, folds: int, seed: int|None, code_variant: str, 
          agent_dir: Path, network: str|None, args_variant: str
          ) -> None:
    """
    Run main benchmark pipline
    """
    logs_dir = (logs_dir / f"{image_name}-{time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())}").resolve()
    runner_spec = RunnerSpec(
        image_name=image_name,
        workers=workers,
        competitions=competitions.resolve(),
        data_dir=Path(data_dir).resolve(),
        runtime_config=runtime_config.resolve(),
        logs_dir=logs_dir,
        log_level=log_level,
        folds=folds,
        seed=seed,
        code_variant=code_variant,
        agent_dir=agent_dir.resolve(),
        network=network,
        extended_schema = (args_variant == "extended")
    )
    setup_logger(runner_spec.log_level, runner_spec.logs_dir, file_log_level=runner_spec.log_level)
    logger.info(f"[blue]Run benchmark with runner spec:\n{runner_spec.model_dump_json(indent=2)}[/blue]")
    run_benchmark(runner_spec)


@cli.command()
def prepare_data():
    """
    Prepare data for benchmark run
    """
    load_data()


if __name__ == "__main__":
    cli()