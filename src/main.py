from .runners import DockerRunner
import bench
from src.bench import *

import os
import asyncio
from .runners import RunnerSpec, Task
from .bench import Competition, Language, CodeLanguage, RunnerInput, RunnerOutput, BenchPipeline
from pathlib import Path
from loguru import logger
from functools import partial
import time

from typing import Any


def merge_results(results: dict, result: dict, idx: int):
    for field in result.keys():
        # For each key in result

        # Check if there is a new key
        if results.get(field) is None:
            if idx > 0:
                results[field] = [None for x in range(idx)]
            else:
                results[field] = []
        else:
            results[field].append(result[field])

    # Append keys which are missing from result
    for field in (set(results.keys()) - set(result.keys())):
        results[field].append(None)


def report(results: dict, result: dict, idx: int, runner: Any, competition: Competition, lang: Language, codelang: CodeLanguage, fold: int | None):
    merge_results(results, {**result, "runner": runner.runner_id, "competition_id": competition.comp_id, "lang": str(lang), "codelang": str(codelang), "fold": fold}, idx)


def report_error(results: dict, idx: int, runner: Any, e: Exception, competition: Competition, lang: Language, codelang: CodeLanguage, fold: int | None):
    print(f"Code execution failed for {runner.runner_id} : {e=}")
    report(results, {"errors": [f"Runner code execution failed : {e=}"], "success": False}, idx, runner, competition, lang, codelang, fold)


async def execute_bench(runner: DockerRunner, max_folds: int) -> None:
    base_path = Path(__file__).resolve().absolute().parent.parent
    bench = BenchPipeline(base_path, max_folds, runner.input_mode == RunnerInput.DescAndData)
    idx = 0
    while (competition := bench.next_competition()) is not None:
    # Process a competition
        bench.prepare_train_data(competition)

        # Iterate over a grid of languages
        for lang in bench.languages():
            for codelang in CodeLanguage:
                success_callback=partial(report, runner=runner, competition=competition, lang=lang, codelang=codelang)
                failure_callback=partial(report_error, runner=runner, competition=competition, lang=lang, codelang=codelang)
                if runner.input_mode == RunnerInput.DescOnly:
                    runner.add_task(idx, bench, competition, lang, codelang, 
                                    [partial(success_callback, idx=idx, fold=None)], 
                                    [partial(failure_callback, idx=idx, fold=None)])
                    idx += 1

                while (fold := bench.next_fold(competition)) is not None:
                    if runner.input_mode == RunnerInput.DescAndData:
                        runner.add_task(idx, bench, competition, lang, codelang, 
                                    [partial(success_callback, idx=idx, fold=fold.idx)], 
                                    [partial(failure_callback, idx=idx, fold=fold.idx)])
                        idx += 1
        results = await runner.run()
        bench.erase_train_data(competition)


def run_benchmark(runner_spec: RunnerSpec) -> None:
    asyncio.run(execute_bench([DockerRunner(runner_spec)], 1))


if __name__ == "__main__":
    run_benchmark()
