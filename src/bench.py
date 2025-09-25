import os
import sys
from enum import StrEnum
import traceback
import json
from pathlib import Path
import shutil
from typing import Any, Optional

import pandas as pd
import importlib
from docker import DockerClient
from loguru import logger

from python.competition import *
from python.splitters import *


class CodeLanguage(StrEnum):
    """Supported programming languages for code generation"""
    Python = "python"
    #R = "rlang"
    #Julia = "julia"


CODEPATHS = {CodeLanguage.Python: "code.py",} #CodeLanguage.R: None, CodeLanguage.Julia: None}
CODE_EXT = {CodeLanguage.Python: ".py"}


class RunnerInput(StrEnum):
    """Types of input that runners can accept"""
    DescOnly = "DescOnly"
    DescAndData = "DescAndData"


class RunnerOutput(StrEnum):
    """Types of output that runners can produce"""
    CodeOnly = "CodeOnly"
    CodeAndData = "CodeAndData"
    DataOnly = "DataOnly"


class BenchMode(StrEnum):
    """Benchmark operation modes"""
    MonolithicPredict = "MONO_PREDICT"
    ModularPredict = "MODULAR_PREDICT"


class BenchPipeline:
    """Main benchmark pipeline for managing competitions and data"""
    def __init__(self, basepath: os.PathLike, max_folds: int = 5, prepare_data: bool = False):
        self.basepath = basepath
        self.max_folds = max_folds
        self.current_comp = 0
        self.current_fold = 0
        self.competitions: list[Competition] = []
        self.folds: dict[str, list[CompetitionData]] = {}
        self._languages: list[Language] = []
        self.grader_module = None
        self.prepare_data = prepare_data

        self._initialize_folders()
        self._load_competitions()
        self._load_graders()

    def _initialize_folders(self) -> None:
        """Initialize folder structure for folds and validation"""
        folds_dir = os.path.join("competitions", "folds")
        private_dir = os.path.join("competitions", "validation")

        if os.path.exists(folds_dir):
            shutil.rmtree(folds_dir)
        if os.path.exists(private_dir):
            shutil.rmtree(private_dir)

    def _load_competitions(self) -> None:
        """Load competitions from competitions.json"""
        comp_json = os.path.join(self.base_path(), "competitions", "competitions.json")
        if not os.path.exists(comp_json):
            print(f"Missing {comp_json} file")
            self.shutdown(1)

        with open(comp_json, "r") as f:
            comp_data = json.load(f)

        tasks_dir = os.path.join(self.base_path(), "competitions", "tasks")


        if os.path.exists(tasks_dir):
            language_files = os.listdir(tasks_dir)
            self._languages = []
            tasks = {}
            
            for file in language_files:
                try:
                    lang = Language(file.split('.')[0])
                    self._languages.append(lang)
                except ValueError:
                    print(f"Bad task file {file}: no such language")
                    self.shutdown(1)

                file_path = os.path.join(tasks_dir, file)
                df = pd.read_csv(file_path)
                tasks[lang] = df.to_dict('records')
        else:
            # Handle tasks directory
            self._languages = [Language.English]
            tasks = {Language.English: []}
            print(f"Note: Tasks directory not found at {tasks_dir}. Using default English setup.")
        
        # Process each competition
        for key, value in comp_data.items():
            if key.startswith("_"):
                continue
            
            comp_tasks = {}
            for lang in self._languages:
                lang_tasks = tasks.get(lang, [])
                comp_task = None
                
                for task in lang_tasks:
                    if task.get("comp-id") == key:
                        comp_task = task
                        break
                
                if comp_task is None:
                    print(f"! warning: competition id {key} : missing in {str(lang)} task descriptions")
                    continue
                
                comp_tasks[lang] = comp_task

            if not comp_tasks:
                print(f"!  warning: competition id {key} : missing in all task descriptions")
                continue

            log_error = lambda x: print(x)
            do_shutdown = lambda x: self.shutdown(x)
            competitions_dir = os.path.join(self.base_path(), "competitions")
            self.competitions.append(Competition(key, value, comp_tasks, competitions_dir, log_error, do_shutdown))
            self.folds[key] = []

    def _load_graders(self) -> None:
        """Load grading functions module"""
        # TODO import this code the usual way
        grader_path = os.path.join(self.base_path(), "python", "grade_functions.py")
        spec = importlib.util.spec_from_file_location("grade_functions", grader_path)
        self.grader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.grader_module)

    def base_path(self) -> os.PathLike:
        return self.basepath

    def shutdown(self, exit_code: int):
        """Shutdown the benchmark with exit code"""
        if exit_code != 0:
            print(f"Benchmark stopped with abnormal exit code {exit_code}")
            traceback.print_exc()
        sys.exit(exit_code)

    def languages(self) -> list[Language]:
        return self._languages

    def total(self) -> int:
        return len(self.competitions)

    def total_folds(self, comp: Competition) -> int:
        return min(self.max_folds, comp.metadata.get("cv_folds", 1))

    def next_competition(self) -> Optional[Competition]:
        """Get next competition"""
        if self.current_comp >= len(self.competitions):
            self.current_comp = 0
            return None
        comp = self.competitions[self.current_comp]
        self.current_comp += 1
        return comp

    def next_fold(self, comp: Competition) -> Optional[CompetitionData]:
        """Get next fold for a competition"""
        if not self.prepare_data or self.current_fold >= self.total_folds(comp):
            self.current_fold = 0
            return None

        fold = self.folds[comp.comp_id][self.current_fold]
        self.current_fold += 1
        return fold

    def test_submission_code(self, comp: Competition, lang: Language, codelang: CodeLanguage, code: str, client: DockerClient, uniq_suf: str, runtime_config: dict[str, Any], image_name: str, extended_schema: bool = False) -> dict:
        """
        Submit the code and return metric
        """

        # Prepare submission dir
        # ======================

        submission_name = f"submission_{uniq_suf}"
        submission_dir = (Path(self.base_path()) / str(codelang) / submission_name).resolve()
        logger.info("SUBMISSION NAME:\n {}", submission_name)
        if os.path.exists(submission_dir):
            shutil.rmtree(submission_dir)
        os.mkdir(submission_dir)

        if codelang == CodeLanguage.Python:
            with open(os.path.join(submission_dir, "__init__.py"), 'w') as f:
                f.write("")

        with open(os.path.join(submission_dir, CODEPATHS[codelang]), 'w') as f:
            f.write(code)

        env_vars = {
            "COMPETITION_ID": comp.comp_id,
            "SUBMISSION_NAME": submission_name,
            "BENCH_LANG": str(lang),
            "BENCH_MODE": str(BenchMode.ModularPredict),
            "EXTENDED_SCHEMA": str(int(extended_schema)),
            "BENCH_FOLDS_OVERRIDE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "/home/bench"
        }
        network_name = "python_no_inet"

        # Check if the network exists
        try:
            client.networks.create(network_name, driver="bridge", internal=True)
        except Exception:
            logger.info("Network already exists, no error")
        container = client.containers.run(
            image=image_name,
            detach=True,
            environment=env_vars,
            volumes={
                submission_dir.as_posix(): {'bind': '/home/bench/submission', 'mode': 'rw'},
                (Path(self.base_path()) / "competitions").resolve().as_posix(): {'bind': '/home/bench/competitions', 'mode': 'ro'}
            },
            network="python_no_inet",
            entrypoint=["mamba", "run", "-n", "agent", "python", "python/bench.py"],
            **runtime_config,
            working_dir="/home/bench"
        )

        exit_code = container.wait(timeout=60*60)
        logs = container.logs().decode('utf-8')

        logger.info("Evaluation container results {}:\n exit_code: {}\n logs: {}", comp.comp_id, exit_code, logs)
        container.remove() 
        results_path = submission_dir / "results.json"
        if not os.path.exists(results_path):
            logger.info("{} container failed to generate output", str(codelang))
            return {"errors": [f"Failed to obtain output for {str(codelang)}"], "success": False}
        with open(results_path, 'r') as f:
            results = json.load(f)
        shutil.rmtree(submission_dir)
        return results

    def test_submission_data(self, comp: Competition, fold: CompetitionData, lang: Language, 
                            codelang: CodeLanguage, data: Any) -> dict:
        """Test submission data with grader"""

        # TODO is this dead code?
        val_dir = os.path.join(self.base_path(), "competitions", "validation", comp.comp_id)
        grader = comp.metadata.get("grader", "default")

        try:
            score = self.grader_module.GRADERS[grader](data, val_dir, comp.metadata)
        except Exception as e:
            err_msg = f"Grader {grader} failed for competition {comp.comp_id} : {e=}"
            print(err_msg)
            return {"manual_submission_score": None, "manual_submission_error": err_msg}

        return {"manual_submission_score": score, "manual_submission_error": None}

    def test_submission_data_path(self, comp: Competition, fold: CompetitionData, lang: Language, 
                                 codelang: CodeLanguage, path_to_data: os.PathLike) -> dict:
        """Test submission data from file path"""
        df = pd.read_csv(path_to_data)
        return self.test_submission_data(comp, fold, lang, codelang, df)

    def prepare_train_data(self, comp: Competition, seed: int) -> None:
        """Prepare training data for all folds"""
        if not self.prepare_data:
            return None

        fold_dir = os.path.join(self.base_path(), "competitions", "folds")
        comp_fold_dir = os.path.join(fold_dir, comp.comp_id)
        private_dir = os.path.join(self.base_path(), "competitions", "validation")
        comp_private_dir = os.path.join(private_dir, comp.comp_id)

        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(private_dir, exist_ok=True)
        os.makedirs(comp_fold_dir, exist_ok=True)
        os.makedirs(comp_private_dir, exist_ok=True)

        log_error = lambda x: print(x)
        do_shutdown = lambda x: self.shutdown(x)

        # Get splitting strategy
        split_strategy = comp.metadata.get("split_strategy", "csv")
        splitter_class = DATA_SPLITTERS.get(split_strategy)
        
        if not splitter_class:
            print(f"Unknown split strategy '{split_strategy}' for competition {comp.comp_id}")
            print(f"Available strategies: {list(DATA_SPLITTERS.keys())}")
            self.shutdown(1)
        
        try:
            splitter = splitter_class(log_error, do_shutdown, False)
            splits = splitter.split_data(comp, self.total_folds(comp))
        except Exception as e:
            print(f"prepare_train_data(): splitting failed: {e=} for competition {comp.comp_id}")
            self.shutdown(1)

        # Prepare additional files
        additional_files = {}
        for file_key, comp_file in comp.get_all_files().items():
            if (file_key not in ["train"] and comp_file.exists() and 
                comp_file.file_type in ["data", "metadata"]):
                additional_files[file_key] = comp_file.path

        # Create folds
        for i, (train_indices, val_indices) in enumerate(splits):
            try:
                train_path, val_path, fold_additional_files = splitter.prepare_fold_data(
                    comp, train_indices, val_indices, i, comp_fold_dir, comp_private_dir
                )
            except Exception as e:
                print(f"prepare_train_data(): fold {i} preparation failed: {e=} for competition {comp.comp_id}")
                self.shutdown(1)

            # Copy additional files
            for file_key, file_path in additional_files.items():
                if os.path.isfile(file_path):
                    fold_file_path = os.path.join(comp_fold_dir, f"{file_key}_{i}{os.path.splitext(file_path)[1]}")
                    shutil.copy2(file_path, fold_file_path)
                    fold_additional_files[file_key] = fold_file_path
                elif os.path.isdir(file_path):
                    fold_dir_path = os.path.join(comp_fold_dir, f"{file_key}_{i}")
                    if not os.path.exists(fold_dir_path):
                        shutil.copytree(file_path, fold_dir_path)
                    fold_additional_files[file_key] = fold_dir_path

            self.folds[comp.comp_id].append(CompetitionData(
                train_path, val_path, i, fold_additional_files
            ))

    def erase_train_data(self, comp: Competition) -> None:
        """Erase training data"""
        if not self.prepare_data:
            return None

        fold_dir = os.path.join(self.base_path(), "competitions", "folds", comp.comp_id)
        private_dir = os.path.join(self.base_path(), "competitions", "validation", comp.comp_id)
        
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
        if os.path.exists(private_dir):
            shutil.rmtree(private_dir)
