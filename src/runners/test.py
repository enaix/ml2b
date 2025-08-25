import bench


class TestRunner:
    input_mode: RunnerInput = RunnerInput.DescOnly
    output_mode: RunnerOutput = RunnerOutput.CodeOnly

    def __init__(self):
        pass

    # run() does not take CompetitionData, since input_mode is DescOnly
    def run(self, bench: BenchPipeline, comp: Competition) -> dict:
        pass

    #def run(self, bench: BenchPipeline, comp: Competition, fold: CompetitionData) -> dict:
    #    pass
