import bench

import runners.test

import os


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


def report(results: dict, result: dict, idx: int, runner: object, competition: Competition, lang: Language, codelang: CodeLanguage, fold: int):
    merge_results(results, {**result, "runner": runner.runner_id, "competition_id": competition.comp_id, "lang": str(lang), "codelang": str(codelang), "fold": fold}, idx)


def report_error(results: dict, idx: int, runner: object, e: Exception, competition: Competition, lang: Language, codelang: CodeLanguage, fold: int):
    print(f"Code execution failed for {runner.runner_id} : {e=}")
    report(results, {"errors": [f"Runner code execution failed : {e=}"], "success": False}, idx, runner, competition, lang, codelang, fold)


def execute_bench(runners: list[object], max_folds: int):
    # Initialize BenchPipeline
    # ...

    # Run the execution loop
    #   if we have runners with output_mode equal to CodeAndData or DataOnly, then we also have to prepare data splits
    #   (just iterate over each competition and each fold if applicable)

    process_data = any([x.input_mode == RunnerInput.DescAndData for x in runners])

    base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
    bench = BenchPipeline(base_path, max_folds, process_data)

    results = {}
    idx = 0

    while (competition := bench.next_competition()) is not None:
        # Process a competition
        bench.prepare_train_data(competition)

        # Iterate over a grid of languages
        for lang in bench.languages():
            for codelang in CodeLanguage:

                # We execute CodeOnly runners first
                for runner in runners:
                    print(f"[{idx}] Executing runner {runner.runner_id}...")
                    if runner.input_mode == RunnerInput.DescOnly:
                        try:
                            result = runner.run(bench, competition, lang, codelang)
                        except BaseException as e:
                            report_error(results, idx, runner, e, competition, lang, codelang, None)
                            idx += 1
                            continue

                        report(results, result, idx, runner, competition, lang, codelang, None)
                        idx += 1


                # Obtain data
                while (fold := bench.next_fold(competition)) is not None:

                    # Execute DescAndData runners
                    for runner in runners:
                        print(f"[{idx}] Executing runner {runner.runner_id} : fold {fold.fold_idx+1}...")

                        if runner.input_mode == RunnerInput.DescAndData:
                            try:
                                result = runner.run(bench, competition, lang, codelang, fold)
                            except BaseException:
                                report_error(results, idx, runner, e, competition, lang, codelang, fold.fold_idx)
                                idx += 1
                                continue

                            report(results, result, idx, runner, competition, lang, codelang, fold.fold_idx)
                            idx += 1

        bench.erase_train_data(competition)


def main():
    # Execute benchmark for only the test runner
    execute_bench([TestRunner()], 1)


if __name__ == "__main__":
    main()
