from src.bench import (
    RunnerInput,
    RunnerOutput,
    BenchPipeline,
    Competition,
    Language,
    CodeLanguage,
)


class DockerRunner:
    input_mode: RunnerInput = RunnerInput.DescOnly
    output_mode: RunnerOutput = RunnerOutput.CodeOnly
    runner_id: str = "test_runner"

    def __init__(self):
        pass

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
