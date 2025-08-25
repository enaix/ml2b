import os
import sys
from enum import Enum
import weakref
import traceback
import json


class Language(Enum):
    English = 1
    # ...


class RunnerInput(Enum):
    DescOnly    = 1  # Runner only takes in competition description
    DescAndData = 2  # Runner takes in competition description and data


class RunnerOutput(Enum):
    CodeOnly    = 1  # Runner returns only the code
    CodeAndData = 2  # Runner returns both code and data
    DataOnly    = 3  # Runner returns only the data


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
    def __init__(self, base_path: os.path.Path, max_folds: int = 5):
        self.base_path = base_path
        self.max_folds = folds
        self.current_comp = 0
        self.current_fold = 0
        self.competitions = []

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
            competitions.append(Competition(key, weakref.ref(self), value, ))
        
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
        pass

    def next_fold(self, comp: Competition) -> CompetitionData:
        pass

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
        # Take the data and do the split
        pass
