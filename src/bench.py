import os
from enum import Enum


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
    def __init__(self):
        # Some competition metadata
        pass

    def get_available_languages(self) -> list[Language]:
        pass

    def get_description(self, lang: Language) -> dict:
        pass


class CompetitionData:
    def __init__(self):
        # Initialize file paths
        pass

    def get_train(self) -> tuple:
        pass

    def get_val(self) -> os.Path:
        pass



class BenchPipeline:
    def __init__(self):
        # TODO configure project root
        pass

    def total(self) -> int:
        """
        Number of remaining competitions
        """
        pass

    def total_folds(self, comp: Competition) -> int:
        pass

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
