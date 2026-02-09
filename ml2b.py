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


DEFAULT_TAG = "runtime-env"
BASE_PATH = Path(__file__).parent
DEFAULT_RUNTIME_PATH = BASE_PATH / "environments" / "runtime"
DEFAULT_CONFIG_PATH = BASE_PATH / "environments" / "container_config.yaml"
DEFAULT_DATA_PATH = BASE_PATH / "competitions" / "data"
DEFAULT_LOGS_PATH = BASE_PATH / "logs"
DEFAULT_COMPETITIONS_PATH = BASE_PATH / "competitions"


@click.group()
def cli():
    pass

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
@click.option(
    "--internet-control",
    type=click.Choice(["no", "proxy"]),
    default="proxy",
    help="Use squid proxy for access to hf and torch domains"
)
@click.option(
    "--proxy-config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    default=DEFAULT_RUNTIME_PATH / "squid.conf",
    help="Path to squid proxy config file, use only if internet-control is proxy",
)
def bench(image_name: str, workers: int, data_dir: Path, 
          runtime_config: Path, log_level: str, logs_dir: Path, 
          competitions: Path, folds: int, seed: int|None, code_variant: str, 
          agent_dir: Path, network: str|None, args_variant: str,
          internet_control: str, proxy_config: Path
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
        extended_schema = (args_variant == "extended"),
        internet_control=internet_control,
        proxy_conf=proxy_config
    )
    setup_logger(runner_spec.log_level, runner_spec.logs_dir, file_log_level=runner_spec.log_level)
    logger.info(f"[blue]Run benchmark with runner spec:\n{runner_spec.model_dump_json(indent=2)}[/blue]")
    run_benchmark(runner_spec)


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
@click.option(
    '--http-proxy',
    type=click.STRING,
    default=None,
    help="HTTP proxy for container build"
)
@click.option(
    '--https-proxy',
    type=click.STRING,
    default=None,
    help="Https proxy for container build"
)
def build_runtime(image_name: str, agent_dir: Path, platform: str, http_proxy: str, https_proxy: str) -> None:
    """
    Build base runtime and agent images
    """
    proxy_settings = {}
    if http_proxy or https_proxy:
        proxy_settings["HTTP_PROXY"] = http_proxy or https_proxy
        proxy_settings["HTTPS_PROXY"] = https_proxy or http_proxy
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
    ), client, proxy_settings)
    if DEFAULT_TAG != image_name:
        build_image(build_args, client, proxy_settings)


HF_DATASET = "enaix/ml2b"
HF_TASKS_DIR = "tasks"
HF_DATA_DIR = "data"

@cli.command()
@click.option(
    "--source",
    type=click.Choice(["huggingface"]),
    default="huggingface",
    help="Dataset source"
)
@click.option("--remove-cache", is_flag=True, help="Remove huggingface hub cache to save space")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_COMPETITIONS_PATH,
    help="Competitions path"
)
@click.option(
    "--hf-dataset",
    type=click.STRING,
    default=HF_DATASET,
    help="Huggingface dataset name"
)
@click.option(
    "--hf-tasks-dir",
    type=click.STRING,
    default=HF_TASKS_DIR,
    help="Tasks dir in hf store"
)
@click.option(
    "--hf-data-dir",
    type=click.STRING,
    default=HF_DATA_DIR,
    help="Data dir in hf store"
)
def prepare_data(remove_cache: bool, source: str, data_dir: Path, hf_dataset: str, hf_tasks_dir: str, hf_data_dir: str):
    """
    Prepare data for benchmark run
    """
    load_data(data_dir, hf_dataset, hf_tasks_dir, hf_data_dir, source, remove_cache)


if __name__ == "__main__":
    cli()
