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


class Language(StrEnum):
    English = "English"
    # ...


class RunnerInput(StrEnum):
    DescOnly    = "DescOnly"     # Runner only takes in competition description
    DescAndData = "DescAndData"  # Runner takes in competition description and data


class RunnerOutput(StrEnum):
    CodeOnly    = "CodeOnly"     # Runner returns only the code
    CodeAndData = "CodeAndData"  # Runner returns both code and data
    DataOnly    = "DataOnly"     # Runner returns only the data


class Competition:
    def __init__(self, comp_id: str, bench: weakref.ref, metadata: dict, comp_data: dict):
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
        self.comp_data = comp_data


    def get_available_languages(self) -> list[Language]:
        return self.comp_data.keys()

    def _get_meta_for_lang(self, lang: Language) -> dict:
        values = self.comp_data.get(lang)
        if values is None:
            print(f"Competition : could not find metadata for language {lang}")
            self.bench().shutdown(1)

    def get_description(self, lang: Language) -> dict:
        return self._get_meta_for_lang(lang).get("description")

    def get_data_card(self, lang: Language) -> dict:
        return self._get_meta_for_lang(lang).get("data_card")

    def get_domain(self, lang: Language) -> dict:
        return self._get_meta_for_lang(lang).get("domain")


class CompetitionData:
    def __init__(self, train_path: Path, val_path: Path, fold_idx: int = 0):
        self.train_path = train_path
        self.val_path = val_path
        self.fold_idx = fold_idx

    def get_train(self) -> os.path.Path:
        return self.train_path

    def get_val(self) -> os.path.Path:
        return self.val_path



class BenchPipeline:
    def __init__(self, base_path: os.path.Path, max_folds: int = 5, prepare_data: bool = False):
        self.base_path = base_path
        self.max_folds = folds
        self.current_comp = 0
        self.current_fold = 0
        self.competitions = []
        self.folds = []
        self.prepare_data = prepare_data

    def _load_competitions(self):
        comp_json = os.path.join(self.base_path(), "competitions", "competitions.json")
        if not os.path.exists(comp_json):
            print(f"Missing {comp_json} file")
            self.shutdown(1)

        with open(comp_json, 'r') as f:
            comp = json.load(f)

        for key, value in comp:
            if key.startswith("_"):
                continue  # skip comments

            # TODO prepare competition csv data
            self.competitions.append(Competition(key, weakref.ref(self), value, ))

    def base_path(self) -> os.path.Path:
        return self.base_path

    def shutdown(self, exit_code: int):
        if exit_code != 0:
            print(f"Benchmark stopped with abnormal exit code {exit_code}")
            traceback.print_exc()
        sys.exit(exit_code)

    def total(self) -> int:
        return len(competitions)

    def total_folds(self, comp: Competition) -> int:
        return min(self.max_folds, comp.metadata.get("cv_folds", 1))

    def next_competition(self) -> Competition:
        """
        Get next competition. This function returns competition info
        """
        if self.current_comp_idx >= len(self.competitions):
            return None
        comp = self.competitions[self.current_comp_idx]
        self.current_comp_idx += 1
        return comp

    def next_fold(self, comp: Competition) -> CompetitionData:
        if not self.prepare_data:
            return None

        # TODO return actual data object
        return None

    def test_submission_code(self, comp: Competition, fold: CompetitionFold, code: str) -> object:
        """
        Submit the code and return metric
        """
        pass

    def test_submission_data(self, comp: Competition, fold: CompetitionFold, data: os.path.Path) -> object:
        """
        Submit the prediction for X_val and return metric
        """
        pass

    def prepare_train_data(self, comp: Competition) -> None:
        """
        Prepare X_train, y_train and X_val files for each fold
        """
        if not self.prepare_data:
            return None

        fold_dir = os.path.join(self.base_path(), "competitions", "folds")
        comp_fold_dir = os.path.join(fold_dir, comp.comp_id)
        fold_dir.mkdir(exist_ok=True)

        if os.path.exists(comp_fold_dir):
            os.rmdir(comp_fold_dir)
        comp_fold_dir.mkdir()

        # Load data
        try:
            train = pd.read_csv(os.path.join("data", competition_id, "train.csv"))
            X, y = train.drop(columns=[comp.metadata["target_col"]]), train[comp.metadata["target_col"]]
        except Exception as e:
            print(f"prepare_train_data() : internal error : data loading failed : {e=} for competition {comp.comp_id}")
            self.shutdown(1)


        kf = KFold(n_splits=self.num_folds(comp))
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            X_train.to_csv(os.path.join(comp_fold_dir, f"X_train_{i}.csv"))
            y_train.to_csv(os.path.join(comp_fold_dir, f"y_train_{i}.csv"))
            X_val.to_csv(os.path.join(comp_fold_dir, f"X_val_{i}.csv"))
            y_val.to_csv(os.path.join(comp_fold_dir, f"y_val_{i}.csv"))

            self.folds[comp.comp_id] = CompetitionData()
            # TODO finish this
