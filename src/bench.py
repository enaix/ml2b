import os
import sys
from enum import Enum, StrEnum
import weakref
import traceback
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import importlib

from typing import Any


class Language(StrEnum):
    English = "English"
    Arab = "Arab"
    Chinese = "Chinese"
    Italian = "Italian"
    Kazach = "Kazach"
    Polish = "Polish"
    Romanian = "Romanian"
    Spanish = "Spanish"
    Turkish = "Turkish"


class CodeLanguage(StrEnum):
    Python = "python"
    R = "rlang"
    Julia = "julia"


CODEPATHS = {CodeLanguage.Python: "code.py", CodeLanguage.R: None, CodeLanguage.Julia: None}


class RunnerInput(StrEnum):
    DescOnly    = "DescOnly"     # Runner only takes in competition description
    DescAndData = "DescAndData"  # Runner takes in competition description and data


class RunnerOutput(StrEnum):
    CodeOnly    = "CodeOnly"     # Runner returns only the code
    CodeAndData = "CodeAndData"  # Runner returns both code and data
    DataOnly    = "DataOnly"     # Runner returns only the data


class Competition:
    def __init__(self, comp_id: str, bench: weakref.ReferenceType, metadata: dict, tasks: dict):
        self.comp_id = comp_id
        self.comp_path = os.path.join(bench().base_path(), "data", comp_id)
        self.train_data = os.path.join(self.comp_path, "train.csv")
        self.test_data = os.path.join(self.comp_path, "test.csv")
        self.metadata = metadata
        self.bench = bench

        if not os.path.exists(self.train_data):
            print(f"Competition {comp_id} : missing train.csv")
            bench().shutdown(1)

        if not os.path.exists(self.test_data):
            print(f"Competition {comp_id} : missing test.csv")
            bench().shutdown(1)

        # Process languages
        self.tasks = tasks


    def get_available_languages(self) -> list[Language]:
        return list(self.tasks.keys())


    def _get_meta_for_lang(self, lang: Language) -> dict:
        values = self.tasks.get(lang)
        if values is None:
            print(f"Competition : could not find metadata for language {lang}")
            self.bench().shutdown(1)
        return values


    def get_description(self, lang: Language) -> dict:
        return self._get_meta_for_lang(lang).get("description")


    def get_data_card(self, lang: Language) -> dict:
        return self._get_meta_for_lang(lang).get("data_card")


    def get_domain(self, lang: Language) -> dict:
        return self._get_meta_for_lang(lang).get("domain")


class CompetitionData:
    def __init__(self, train_path: os.PathLike, val_path: os.PathLike, fold_idx: int = 0):
        self.train_path = train_path
        self.val_path = val_path
        self.fold_idx = fold_idx


    def get_train(self) -> os.PathLike:
        return self.train_path


    def get_val(self) -> os.PathLike:
        return self.val_path



class BenchPipeline:
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
        folds_dir = os.path.join("competitions", "folds")
        private_dir = os.path.join("competitions", "private")

        if os.path.exists(folds_dir):
            os.rmdir(folds_dir)
        if os.path.exists(private_dir):
            os.rmdir(private_dir)


    def _load_competitions(self) -> None:
        comp_json = os.path.join(self.base_path(), "competitions", "competitions.json")
        if not os.path.exists(comp_json):
            print(f"Missing {comp_json} file")
            self.shutdown(1)

        with open(comp_json, 'r') as f:
            comp = json.load(f)

        tasks_dir = os.path.join(self.base_path(), "competitions", "tasks")
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

            df = pd.read_csv(file)
            tasks[lang] = df.to_json()

        for key, value in comp:
            if key.startswith("_"):
                continue  # skip comments

            comp_tasks = {}
            for lang in self._languages:
                # Try to find the matching competition id
                if not tasks[lang].get("comp-id"):
                    print(f"{str(lang)} task descriptions : missing comp-id field")
                    self.shutdown(1)

                try:
                    idx = tasks[lang]["comp-id"].index(key)
                except ValueError:
                    print(f"! warning: competition id {key} : missing in {str(lang)} task descriptions")
                    continue

                comp_tasks[lang] = {key : value[idx] for (key, value) in tasks[lang].items()}

            if not comp_tasks:
                print("   ----")
                print(f"!  warning: competition id {key} : missing in all task descriptions")
                continue

            self.competitions.append(Competition(key, weakref.ref(self), value, comp_tasks))
            self.folds[key] = []
            # Creation and deletion of training data should be handled from the main loop
            # self.prepare_train_data(self.competitions[-1])


    def _load_graders(self) -> None:
        self.grader_module = importlib.import_module(os.path.join(self.base_path(), "python", "grade_functions.py"))


    def base_path(self) -> os.PathLike:
        return self.basepath


    def shutdown(self, exit_code: int):
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


    def next_competition(self) -> Competition | None:
        """
        Get next competition. This function returns competition info
        """
        if self.current_comp >= len(self.competitions):
            self.current_comp = 0
            return None
        comp = self.competitions[self.current_comp]
        self.current_comp += 1
        return comp


    def next_fold(self, comp: Competition) -> CompetitionData | None:
        """
        Note: this function does not keep track of competition id!
        """
        if not self.prepare_data or self.current_fold >= self.total_folds(comp):
            self.current_fold = 0
            return None

        fold = self.folds[comp.comp_id][self.current_fold]
        self.current_fold += 1
        return fold


    def test_submission_code(self, comp: Competition, lang: Language, codelang: CodeLanguage, code: str) -> dict:
        """
        Submit the code and return metric
        """

        # Prepare submission dir
        # ======================

        submission_dir = os.path.join(str(codelang), "submission")
        if os.path.exists(submission_dir):
            os.rmdir(submission_dir)
        os.mkdir(submission_dir)

        if codelang == CodeLanguage.Python:
            with open(os.path.join(submission_dir, "__init__.py"), 'w') as f:
                f.write("")
        
        with open(os.path.join(submission_dir, CODEPATHS[codelang]), 'w') as f:
            f.write(code)


        result = subprocess.run(
                ["docker", "compose", "run", str(codelang)],
                capture_output=True,
                text=True,
                env={"COMPETITION_ID": comp.comp_id, "BENCH_LANG": str(lang), "BENCH_MODE": str(RunnerOutput.CodeOnly)},
                timeout=60*60  # 1 hour timeout
            )
        if result.returncode != 0:
            print(f"{str(codelang)} container execution failed: {result.stderr}")
            self.shutdown(1)

        results_path = os.path.join(self.base_path(), str(codelang), "submission")
        if not os.path.exists(results_path):
            print(f"{str(codelang)} container failed to generate output")
            return {"errors": [f"Failed to obtain output for {str(codelang)}"], "success": False}

        with open(results_path, 'r') as f:
            results = json.load(f)
        return results


    def test_submission_data(self, comp: Competition, fold: CompetitionData, lang: Language, codelang: CodeLanguage, data: Any) -> dict:
        """
        Submit the prediction for X_val and return results. These results need to be merged with submission code results
        """
        val = os.path.join(self.base_path(), "competitions", "validation", comp.comp_id)
        grader = comp.metadata.get("grader", "default")

        try:
            score = self.grader_module.GRADERS[grader](data, val, comp.metadata)
        except Exception as e:
            err_msg = f"Grader {grader} failed for competition {comp} : {e=}"
            print(err_msg)
            return {"manual_submission_score": None, "manual_submission_error": err_msg}

        return {"manual_submission_score": score, "manual_submission_error": None}


    def test_submission_data_path(self, comp: Competition, fold: CompetitionData, lang: Language, codelang: CodeLanguage, path_to_data: os.PathLike) -> dict:
        """
        Submit the prediction for X_val and return metric. Data is provided as a CSV file
        """
        df = pd.read_csv(path_to_data)
        return self.test_submission_data(comp, fold, lang, codelang, df)


    def prepare_train_data(self, comp: Competition) -> None:
        """
        Prepare X_train, y_train and X_val files for each fold
        """
        if not self.prepare_data:
            return None

        fold_dir = os.path.join(self.base_path(), "competitions", "folds")
        comp_fold_dir = os.path.join(fold_dir, comp.comp_id)
        private_dir = os.path.join(self.base_path(), "competitions", "validation")
        comp_private_dir = os.path.join(private_dir, comp.comp_id)

        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)
        if not os.path.exists(private_dir):
            os.mkdir(private_dir)

        os.mkdir(comp_fold_dir)
        os.mkdir(comp_private_dir)

        # Load data
        try:
            train = pd.read_csv(os.path.join("data", comp.comp_id, "train.csv"))
            X, y = train.drop(columns=[comp.metadata["target_col"]]), train[comp.metadata["target_col"]]
        except Exception as e:
            print(f"prepare_train_data() : internal error : data loading failed : {e=} for competition {comp.comp_id}")
            self.shutdown(1)


        kf = KFold(n_splits=self.total_folds(comp))
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            train = pd.concat([X_train, y_train], axis=1)
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            train_path = os.path.join(comp_fold_dir, f"train_{i}.csv")
            x_val_path = os.path.join(comp_fold_dir, f"X_val_{i}.csv")
            y_val_path = os.path.join(comp_private_dir, f"y_val_{i}.csv")
            train.to_csv(train_path, index=False)
            X_val.to_csv(x_val_path, index=False)
            y_val.to_csv(y_val_path, index=False)  # this file remains private
            self.folds[comp.comp_id].append(CompetitionData(train_path, x_val_path, i))


    def erase_train_data(self, comp: Competition) -> None:
        """
        Erase training data which is no longer needed
        """
        if not self.prepare_data:
            return None

        fold_dir = os.path.join(self.base_path(), "competitions", "folds", comp.comp_id)
        private_dir = os.path.join(self.base_path(), "competitions", "validation", comp.comp_id)
        os.rmdir(fold_dir)
        os.rmdir(private_dir)
