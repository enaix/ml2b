from src.bench import (
    RunnerInput,
    RunnerOutput,
    BenchPipeline,
    Competition,
    Language,
    CodeLanguage,
    CompetitionData,
    CODE_EXT
)
from src.task_builder import TaskBuilder, TaskContext, TaskDescription
import time
import docker
from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path
from typing import Any, Callable
import asyncio
import traceback
from typing import Iterable, Literal
import uuid
from loguru import logger
from docker.models.containers import Container
import tempfile
import yaml
import json
import tarfile
import shutil
import warnings
import re
import os
warnings.filterwarnings("ignore", category=ResourceWarning)
from loaders import DATA_LOADERS, DataLoader

class RunnerSpec(BaseModel):
    """
    Spec for running competition containers
    """
    image_name: str
    workers: int
    competitions: Path
    data_dir: Path
    runtime_config: Path
    logs_dir: Path
    log_level: Literal["TRACE", "DEBUG", "INFO", "WARNING"]
    folds: int
    seed: int|None
    code_variant: Literal["extended", "short"]
    agent_dir: Path
    network: str | None
    extended_schema: bool

class Task(BaseModel):
    idx: int
    bench: BenchPipeline
    competition: Competition
    lang: Language
    codelang: CodeLanguage
    fold: CompetitionData | None = None
    success_callbacks: Iterable[Callable[[Any], Any]]
    failure_callbacks: Iterable[Callable[[Any], Any]]

    @property
    def unique_name(self):
        fold = "only_code"
        if self.fold is not None:
            fold = f"{self.fold.fold_idx}"
        task_name = f"{self.competition.comp_id}-{self.lang}-{self.codelang}-{fold}"
        return task_name
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TaskOut(BaseModel):
    success: bool = False
    code: str | None = None
    submission: str | None = None
    error: str | None = None
    callbacks_results: list[Any] = Field(default_factory=list)
    consumed_time: float = 0.0
    code_results: dict | None = None

class TasksManager(BaseModel):
    results: dict[int, TaskOut] = Field(default_factory=dict)
    consumed_time: float = 0.0

def create_temp_file_with_text(text: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    file_path = tmp_dir / "instructions.txt"
    file_path.write_text(text, encoding="utf-8")
    return file_path

class AgentSpec(BaseModel):
    env_vars: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    agent_dir: Path
    start_script: Path
    kwargs_type: Literal["omegaconf", "argparse"]

def get_env(value: str) -> str|None:
    """Returns the name of the environment variable in the format `${secrets.<name>}`."""

    if not isinstance(value, str):
        return None

    env_var_pattern = r"\$\{\{\s*secrets\.(\w+)\s*\}\}"
    match = re.match(env_var_pattern, value)

    if not match:
        return None

    return match.group(1)

def parse_env_vars(vars: dict[str, Any]) -> dict[str, Any]:
    for key, var in vars.items():
        evar_name = get_env(var)
        if evar_name is not None:
            evar = os.getenv(evar_name)
            if evar is None:
                raise ValueError(f"ENV {evar_name} not set")
            vars[key] = evar
    return vars


def build_agent_spec(agent_dir: Path) -> AgentSpec:
    agent_dir = agent_dir.resolve()
    with open(agent_dir / "config.yaml", "r") as f:
        content = yaml.safe_load(f)
    kwargs = parse_env_vars(content.get("kwargs", {}))
    env_vars = parse_env_vars(content.get("env_vars", {}))
    kwargs_type = content.get("kwargs_type", "argparse")
    agent_spec = AgentSpec(
        env_vars=env_vars, 
        kwargs=kwargs,
        agent_dir=agent_dir,
        start_script=agent_dir / content.get("start"),
        kwargs_type=kwargs_type
        )
    logger.info("Agent spec: {}", agent_spec.model_dump_json(indent=2))
    return agent_spec   

def parse_runtime_config(config_path: Path) -> dict[str, str]:
    with open(config_path, "r") as fp:
        config = json.load(fp)
    proc_config = {k: v for k, v in config.items() if k != "gpus"}

    if "gpus" in config and config["gpus"] != 0:
        gpu_count = config["gpus"]
        proc_config["device_requests"] = [
            docker.types.DeviceRequest(count=gpu_count, capabilities=[["gpu"]])
        ]

    proc_config["nano_cpus"] = int(proc_config["nano_cpus"]) if "nano_cpus" in proc_config else None
    return proc_config

class DockerRunner:
    input_mode: RunnerInput = RunnerInput.DescAndData
    output_mode: RunnerOutput = RunnerOutput.CodeOnly
    runner_id: str = "docker_runner"

    def __init__(self, runner_spec: RunnerSpec):
        self.runner_spec = runner_spec
        self.workers = []
        self.tasks: asyncio.Queue[Task] = asyncio.Queue()
        self.tasks_manager: TasksManager | None = None
        self.client = docker.from_env()
        self.stop_event = asyncio.Event()
        self.task_builder = TaskBuilder()
        self.agent_spec = build_agent_spec(self.runner_spec.agent_dir)
    
    def add_task(self, task: Task) -> None:
        self.tasks.put_nowait(task)

    def create_container(self, task: Task, instructions_file: str) -> Container:
        volumes = {
            Path(task.fold.train_path).resolve().as_posix(): {
                "bind": "/home/data",
                "mode": "ro"
            },
            Path(instructions_file).resolve().as_posix(): {
                "bind": "/home/instructions.txt",
                "mode": "ro" 
            }
        }
        container_uuid = str(uuid.uuid4().hex)
        time_id = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
        container = self.client.containers.create(
            image=self.runner_spec.image_name,
            name=f"task-{task.competition.comp_id}-{time_id}-{container_uuid}",
            detach=True,
            volumes=volumes,
            environment=self.agent_spec.env_vars,
            **parse_runtime_config(self.runner_spec.runtime_config),
            network=self.runner_spec.network
        )
        logger.info("[blue]Created: {}[/blue]", container.name)
        return container

    def clean_up_container(self, container: Container) -> None:
        try:
            container.stop()
            container.remove()
            logger.info("[blue]Container {} stopped and removed.[/blue]", container.name)
        except Exception as e:
            logger.error(
                "Error removing {}\n{}",
                container.name, e
            )
    
    def execute(self, container: Container, task: Task) -> None:
        #need rewrite for usability
        command = ["bash", "agent/start.sh"]
        if self.agent_spec.kwargs_type == "argparse":
            for key, value in self.agent_spec.kwargs.items():
                command += [f"--{key}", str(value)]
        elif self.agent_spec.kwargs_type == "omegaconf":
            command += [f"{key}={value}" for key, value in self.agent_spec.kwargs.items()]
        else:
            raise ValueError("Not the right kwargs type, use (omegaconf/argparse)")
        logger.info("Run command: {}", command)
        exit_code, output = container.exec_run(command, stream=True, user="nonroot")
        for chunk in output:
            logger.info("[yellow]Container log {}[/yellow]\n {}", task.unique_name,  chunk.decode('utf-8').strip())

    def load_file(self, container: Container, load_dir: str, log_dir: Path) -> None:
        log_dir.mkdir(exist_ok=True, parents=True)
        try:
            stream, _ = container.get_archive(load_dir)
            tmp_tar_path = log_dir / "tmp.tar"

            with open(tmp_tar_path, "wb") as f:
                for chunk in stream:
                    f.write(chunk)

            with tarfile.open(tmp_tar_path, "r") as tar:
                for member in tar.getmembers():
                    if member.issym() or member.islnk():
                        try:
                            target = tar.extractfile(member.linkname)
                        except KeyError:
                            continue
                        if target:
                            out_path = log_dir / member.name
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(out_path, "wb") as f:
                                f.write(target.read())
                    else:
                        tar.extract(member, path=log_dir)

            tmp_tar_path.unlink()
        except Exception as e:
            logger.error(f"Error extracting output file: {e}")


    def extract_artifacts(self, container: Container, log_dir: str|Path) -> None:
        #need to implement more flexible path extraction
        for load_dir in ["/home/logs", "/home/submission", "/home/artifacts"]:
            self.load_file(container, load_dir, log_dir)

    def container_runtime(self, task: Task, task_out: TaskOut):
        loader_name = task.competition.metadata.get("load_strategy", "default")
        loader_class: DataLoader  = DATA_LOADERS.get(loader_name)
    
        task_description = TaskDescription(
            description=task.competition.get_description(task.lang),
            domain=task.competition.get_domain(task.lang),
            metric=task.competition.get_metric(task.lang),
            datacard=task.competition.get_data_card(task.lang)
        )
        task_context = TaskContext(
            code_lang_extention=CODE_EXT[task.codelang],
            code_lang=task.codelang,
            competition_type_code=(self.output_mode == RunnerOutput.CodeOnly or self.output_mode == RunnerOutput.CodeAndData),
            competition_type_file=(self.output_mode != RunnerOutput.CodeOnly),
            code_template_variant=self.runner_spec.code_variant,
            task_info=task_description,
            full_schema=loader_class.schema(expose=self.runner_spec.extended_schema),
            schema_dict=loader_class.schema_dict(expose=self.runner_spec.extended_schema)
        )
        task_prompt = self.task_builder.render(task_context)
        logger.info("TASK: {}", task_prompt)
        container = None
        instructions_file = None
        time_start = time.monotonic()
        try:
            instructions_file = create_temp_file_with_text(task_prompt)
            container = self.create_container(task, instructions_file)

            container.start()
            self.execute(container, task)
            self.extract_artifacts(container, Path(self.runner_spec.logs_dir / task.unique_name).resolve())
        except Exception as e:
            raise e
        finally:
            if container is not None:
                self.clean_up_container(container)
            if instructions_file.parent.exists():
                shutil.rmtree(instructions_file.parent)
        task_out.consumed_time = time.monotonic() - time_start
        return task_out

    def evaluate_results(self, task: Task, task_out: TaskOut):
        run_dir = Path(self.runner_spec.logs_dir / task.unique_name, encoding="utf-8").resolve()
        if self.output_mode == RunnerOutput.CodeOnly:
            code_file = run_dir / "submission" / "submission.py"
            code = code_file.read_text(encoding="utf-8")
            result = task.bench.test_submission_code(
                    task.competition,
                    task.lang,
                    task.codelang,
                    code,
                    self.client,
                    task.unique_name,
                    parse_runtime_config(self.runner_spec.runtime_config),
                    self.runner_spec.image_name,
                    self.runner_spec.extended_schema
                )
            with open(run_dir / "code_results.json", "w", encoding="utf-8") as fp:
                json.dump(result, fp, indent=2)


    async def task_router(self) -> None:
        while True:
            try:
                task: Task = await asyncio.wait_for(self.tasks.get(), timeout=1)
            except asyncio.TimeoutError:
                if self.stop_event.is_set() and self.tasks.empty():
                    break
                continue
            task_out = TaskOut()
            try:
                logger.info("[blue]Start\n fold idx: {}\n task: {}\n language: {}\n code language: {}[/blue]", task.fold.fold_idx, task.competition.comp_id, task.lang, task.codelang)
                await asyncio.to_thread(
                    self.container_runtime,
                    task=task,
                    task_out=task_out
                    )
                task_out.success = True
                logger.success("[green]Task idx: {} finish sucessfuly[/green]", task.idx)
                logger.info("[blue]Start evaluation\n fold idx: {}\n task: {}\n language: {}\n code language: {}[/blue]", task.fold.fold_idx, task.competition.comp_id, task.lang, task.codelang)
                await asyncio.to_thread(
                    self.evaluate_results,
                    task=task,
                    task_out=task_out
                    )
                for cb in task.success_callbacks:
                    cb({})
            except Exception:
                for cb in task.failure_callbacks:
                    cb({})
                trace = traceback.format_exc()
                task_out.error = trace
                logger.error("[red]Task idx: {} crashes with error\n{}[/red]", task.idx, trace)
            finally:
                self.tasks_manager.results[task.idx] = task_out
                self.tasks.task_done()


    async def run(
        self,
    ) -> dict[str, Any]:
        self.tasks_manager = TasksManager(consumed_time=time.monotonic())
        for i in range(self.runner_spec.workers):
            worker = asyncio.create_task(self.task_router())
            self.workers.append(worker)
        await asyncio.gather(*self.workers)
        self.tasks_manager.consumed_time = time.monotonic() - self.tasks_manager.consumed_time
        return self.tasks_manager.model_dump()
