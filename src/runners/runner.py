from src.bench import (
    RunnerInput,
    RunnerOutput,
    BenchPipeline,
    Competition,
    Language,
    CodeLanguage,
)
import docker
from pydantic import BaseModel
from pathlib import Path

class RunnerSpec(BaseModel):
    """
    Spec for running competition containers
    """
    agent_name: str
    num_workers: int
    competition_set: Path
    data_dir: Path
    container_config: Path

class DockerRunner:
    input_mode: RunnerInput = RunnerInput.DescOnly
    output_mode: RunnerOutput = RunnerOutput.CodeOnly
    runner_id: str = "docker_runner"
    client: docker.DockerClient | None = None

    def __init__(self, runner_spec: RunnerSpec):
        self.runner_spec = runner_spec

    # run() does not take CompetitionData, since input_mode is DescOnly
    def run(
        self,
        bench: BenchPipeline,
        comp: Competition,
        lang: Language,
        codelang: CodeLanguage,
    ) -> dict:
        # get description and other stuff from comp
        # call bench to execute
        # return resulting score
        pass

    # if we needed to process data
    # def run(self, bench: BenchPipeline, comp: Competition, fold: CompetitionData) -> dict:
    #    pass
