from .runners import DockerRunner
from src.bench import *
import asyncio
from .runners import RunnerSpec, Task
from .bench import Competition, Language, CodeLanguage, RunnerInput, BenchPipeline
from pathlib import Path
from loguru import logger
from functools import partial
from collections import defaultdict
from typing import Any
import traceback
import docker
from os import PathLike

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


async def feed_competitions(bench: BenchPipeline, runner: DockerRunner):
    active_tasks = defaultdict(int)
    comp_by_id = {}
    all_done = asyncio.Event()

    
    def all_done_callback(*args, comp_id, **kwargs) -> None:
        active_tasks[comp_id] -= 1
        if active_tasks[comp_id] == 0:
            bench.erase_train_data(comp_by_id[comp_id])
            if all(v == 0 for v in active_tasks.values()):
                all_done.set()

    idx = 0
    while (competition := bench.next_competition()) is not None:
            comp_by_id[competition.comp_id] = competition
            try:
                await asyncio.to_thread(bench.prepare_train_data, competition, runner.runner_spec.seed)
            except BaseException:
                logger.info("Error preeparing data: {}", traceback.format_exc())
                continue

            for lang in bench.languages():
                for codelang in CodeLanguage:
                    logger.debug(f"Add to queue\n task: {competition.comp_id}\n mode: {runner.input_mode}\n language: {lang}\n code language: {codelang}")
                    if runner.input_mode == RunnerInput.DescOnly:
                        task = Task(
                            idx=idx,
                            bench=bench,
                            competition=competition,
                            lang=lang,
                            codelang=codelang,
                            success_callbacks=[partial(all_done_callback, comp_id=competition.comp_id)],
                            failure_callbacks=[partial(all_done_callback, comp_id=competition.comp_id)]
                        )
                        active_tasks[competition.comp_id] += 1
                        runner.add_task(task)
                        idx+=1

                    while (fold := bench.next_fold(competition)) is not None:
                        if runner.input_mode == RunnerInput.DescAndData:
                            active_tasks[competition.comp_id] += 1
                            task = Task(
                                idx=idx,
                                bench=bench,
                                competition=competition,
                                lang=lang,
                                codelang=codelang,
                                fold=fold,
                                success_callbacks=[partial(all_done_callback, comp_id=competition.comp_id)],
                                failure_callbacks=[partial(all_done_callback, comp_id=competition.comp_id)]
                            )
                            runner.add_task(task)
                            idx += 1
    await all_done.wait()
    runner.stop_event.set()

async def execute_bench(runner_spec: RunnerSpec):
    base_path = Path(__file__).resolve().absolute().parent.parent
    runner = DockerRunner(runner_spec)
    bench = BenchPipeline(base_path, runner_spec.folds, runner.input_mode == RunnerInput.DescAndData)
    
    feed_task = asyncio.create_task(feed_competitions(bench, runner))
    run_task = asyncio.create_task(runner.run())
    
    try:
        await asyncio.gather(feed_task, run_task, return_exceptions=True)
    except Exception as e:
        logger.error("Benchmark failed: {}", e)
        runner.stop_event.set()
        raise
    finally:
        runner.stop_event.set()


def setup_proxy(proxy_conf: PathLike):
    """
    Setup and run proxy container for grading with internet restrictions
    """
    client = docker.from_env()
    proxy_conf = Path(proxy_conf).resolve()
    if not proxy_conf.exists():
        raise FileNotFoundError(f"squid.conf not found at {proxy_conf}")
    
    try:
        client.networks.create("benchmark_net", driver="bridge")
        logger.info("Created network 'benchmark_net'")
    except docker.errors.APIError as e:
        logger.info("{}", str(e))
    
    try:
        proxy = client.containers.get("benchmark-proxy")
        if proxy.status != "running":
            proxy.start()
            logger.info("Started existing Squid proxy")
        else:
            logger.info("Squid proxy already running")
    except docker.errors.NotFound:
        proxy = client.containers.run(
            "ubuntu/squid:latest",
            name="benchmark-proxy",
            detach=True,
            network="benchmark_net",
            volumes={
                proxy_conf.as_posix(): {
                    "bind": "/etc/squid/squid.conf",
                    "mode": "ro"
                }
            },
            restart_policy={"Name": "unless-stopped"}
        )
        logger.info("Started new Squid proxy")
    return proxy

def cleanup_proxy():
    """Stop proxy container"""
    client = docker.from_env()
    try:
        proxy = client.containers.get("benchmark-proxy")
        proxy.stop()
        proxy.remove()
        logger.info("Cleaned up proxy container")
    except docker.errors.NotFound:
        logger.info("Proxy container not found")


def run_benchmark(runner_spec: RunnerSpec) -> None:
    if runner_spec.internet_control == "proxy":
        setup_proxy(runner_spec.proxy_conf)
    asyncio.run(execute_bench(runner_spec))
    cleanup_proxy()

