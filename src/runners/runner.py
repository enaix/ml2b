from src.bench import (
    RunnerInput,
    RunnerOutput,
    BenchPipeline,
    Competition,
    Language,
    CodeLanguage,
)
import time
import docker
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Callable
import asyncio
import traceback
from typing import Any, Iterable
import uuid
from loguru import logger
from docker.models.containers import Container


class RunnerSpec(BaseModel):
    """
    Spec for running competition containers
    """
    image_name: str
    num_workers: int
    competition_set: Path
    data_dir: Path
    container_config: Path

class Task(BaseModel):
    idx: int
    bench: BenchPipeline
    competition: Competition
    lang: Language
    codelang: CodeLanguage
    fold: int | None = None
    success_callbacks: Iterable[Callable[[Any], Any]]
    failure_callbacks: Iterable[Callable[[Any], Any]]

class TaskOut(BaseModel):
    success: bool = False
    code: str | None
    submission: str | None
    error: str | None
    callbacks_results: Iterable[Any]
    consumed_time: float

class TasksManager(BaseModel):
    results: dict[int, TaskOut]
    consumed_time: float

class DockerRunner:
    input_mode: RunnerInput = RunnerInput.DescOnly
    output_mode: RunnerOutput = RunnerOutput.CodeOnly
    runner_id: str = "docker_runner"

    def __init__(self, runner_spec: RunnerSpec):
        self.runner_spec = runner_spec
        self.workers = []
        self.tasks: asyncio.Queue[Task] = asyncio.Queue()
        self.tasks_manager: TasksManager | None = None
        self.client = docker.from_env()
    
    def add_task(self,
                 idx: int, 
                 bench: BenchPipeline, 
                 competition: Competition, 
                 lang: Language, 
                 codelang: CodeLanguage,
                 success_callbacks: Iterable[Callable[[Any], Any]],
                 failure_callbacks: Iterable[Callable[[Any], Any]]
                 ) -> None:
        task = Task(
            idx=idx,
            bench=bench,
            competition=competition,
            lang=lang,
            codelang=codelang,
            success_callbacks=success_callbacks,
            failure_callbacks=failure_callbacks
        )
        self.tasks.put_nowait(task)

    def create_container(self, task: Task, volumes: dict[Path, dict[str, str]]) -> Container:
        container_uuid = str(uuid.uuid4().hex)
        time_id = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
        container = self.client.containers.create(
            image=self.runner_spec.agent_name,
            name=f"task-{task.competition.comp_id}-{time_id}-{container_uuid}",
            detach=True,
            volumes=volumes,
            # environment=...,
        )
        logger.info(f"[blue]Created: {container.name}[/blue]")
        return container

    def execute_container(self, container: Container):
        cmd = ["bash", "/home/agent/start.sh"]
        logger.info(f"[green]Running agent: {container.name}...[/green]")
        exit_code, out = container.exec_run(cmd, stream=True, user="nonroot")
        for chunk in out:
            logger.info(f"[yellow]Container {chunk.decode('utf-8').strip()}[/yellow]")

    def load_from_container(self, container: Container, path_inside: str | Path):
        try:
            stream, _ = container.get_archive(path_inside)
            tmp_tar_path = local_dir / "tmp.tar"

            with open(tmp_tar_path, "wb") as f:
                for chunk in stream:
                    f.write(chunk)


            with tarfile.open(tmp_tar_path, "r") as tar:
                tar.extractall(path=local_dir)

            tmp_tar_path.unlink()
        except FileNotFoundError:
            logger.warning(f"Nothing found in container at {container_file_path}.")
        except Exception as e:
            logger.error(f"Error extracting output file: {e}")

    def extract_artifacts(self, container: Container):
        
        dirs = ['/home/logs', "/home/submission", "/home/artifacts"]
        for dir in dirs:
            self.load_from_container(container, dir)
            ...

    def runtime_worker(self, task: Task, task_out: TaskOut):
        volumes = {
            Path(task.competition.train_data).resolve().as_posix(): {
                "bind": "/home/data",
                "mode": "ro"
            }
        }

        container = self.create_container(task, volumes=volumes)
        logger.info(f"[blue]Run: {container.name}[/blue]")
        try:
            start_time = time.monotonic()
            container.start()
            self.execute_container(container)
            self.save_artifacts()
            task_out.consumed_time = time.monotonic() - start_time
            logger.info(f"[blue]Running: {container.name} spent {task_out.consumed_time}[/blue]")
            return None
        except Exception as e:
            raise e
        finally:
            clean_up()

    async def task_router(self) -> None:
        while True:
            task = await self.tasks.get()
            #initialize loger
            task_out = TaskOut()
            try:
                await asyncio.to_thread(
                    self.runtime_worker,
                    task=task
                    )
                task_out.success = True
            except Exception as e:
                trace = traceback.format_exc()
                task_out.error = trace
            finally:
                self.tasks.task_done()
                self.tasks_manager[task.idx] = task_out

    async def run(
        self,
    ) -> dict[str, Any]:
        self.tasks_manager = TasksManager()
        for _ in range(self.runner_spec.num_workers):
            worker = asyncio.create_task(self.runtime_worker)
            self.workers.append(worker)
        start_time = time.monotonic()
        await self.tasks.join()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.tasks_manager.consumed_time = time.monotonic() - start_time
        self.tasks.clear()
        return self.tasks_manager.model_dump()
