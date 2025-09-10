from enum import StrEnum
import os



class FileTypes(StrEnum):
    """CompetitionFile type by extension"""
    Data     = "data"
    Metadata = "metadata"
    Other    = "other"


class CompetitionFile:
    """Represents a single file in a competition"""
    def __init__(self, name: str, path: str, file_type: str = "data", required: bool = True):
        self.name = name
        self.path = path
        self.file_type = file_type
        self.required = required

    def exists(self) -> bool:
        return os.path.exists(self.path)


class Competition:
    def __init__(self, comp_id: str, metadata: dict, tasks: dict, competitions_dir: os.PathLike, log_error: any, do_shutdown: any):
        """
        Represents a machine learning competition.

        Args:
            comp_id: Unique competition identifier
            metadata: Competition configuration and metadata
            tasks: Task descriptions for different languages
            competitions_dir: Directory where competitions are stored
            log_error: Function which logs error to stdout or some other interface
            do_shutdown: Shutdown the application on fatal error with an error code
        """
        self.comp_id = comp_id
        self.metadata = metadata
        self.tasks = tasks
        self.competitions_dir = competitions_dir
        self.log_error = log_error
        self.do_shutdown = do_shutdown
        self.files = None

        self.comp_path = os.path.join(self.competitions_dir, "data", comp_id)


    def set_files(self, files) -> None:
        """Set competition files"""
        self.files = files

    def get_file(self, file_key: str) -> Optional[CompetitionFile]:
        """Get a specific file by key"""
        return self.files.get(file_key)

    def get_files_by_type(self, file_type: str) -> List[CompetitionFile]:
        """Get all files of a specific type"""
        return [f for f in self.files.values() if f.file_type == file_type]

    def get_all_files(self) -> Dict[str, CompetitionFile]:
        """Get all files in the competition"""
        return self.files.copy()

    def get_data_files(self) -> Dict[str, str]:
        """Get all data files as a dict of {name: path}"""
        data_files = {}
        for file_key, comp_file in self.files.items():
            if comp_file.file_type in ["data", "metadata"] and comp_file.exists():
                data_files[file_key] = comp_file.path
        return data_files

    def get_data_loader(self, loader_name: Optional[str] = None) -> DataLoader:
        """Get the appropriate data loader for this competition"""
        # TODO remove this code
        if loader_name is None:
            loader_name = self.metadata.get('data_loader', 'default')

        loader_class = DATA_LOADERS.get(loader_name, DefaultDataLoader)
        return loader_class()

    def get_available_languages(self) -> list[Language]:
        """Get available languages for this competition"""
        return list(self.tasks.keys())

    def _get_meta_for_lang(self, lang: Language) -> dict:
        """Get metadata for a specific language"""
        values = self.tasks.get(lang)
        if values is None:
            print(f"Competition: could not find metadata for language {lang}")
            self.bench().shutdown(1)
        return values

    def get_description(self, lang: Language) -> dict:
        """Get description for a specific language"""
        return self._get_meta_for_lang(lang).get("description")

    def get_data_card(self, lang: Language) -> dict:
        """Get data card for a specific language"""
        return self._get_meta_for_lang(lang).get("data_card")

    def get_domain(self, lang: Language) -> dict:
        """Get domain information for a specific language"""
        return self._get_meta_for_lang(lang).get("domain")

    def get_metric(self, lang: Language) -> dict:
        return self._get_meta_for_lang(lang).get("metric")

    def get_code_ext(self, code_lang: CodeLanguage) -> str:
        return CODE_EXT[code_lang]

    # Legacy properties for backward compatibility
    # TODO remove this code (after checking src/bench.py)
    @property
    def train_data(self) -> str:
        """Get train data path (backward compatibility)"""
        train_file = self.get_file("train")
        return train_file.path if train_file else os.path.join(self.comp_path, "train.csv")

    @property
    def test_data(self) -> str:
        """Get test data path (backward compatibility)"""
        test_file = self.get_file("test")
        return test_file.path if test_file else os.path.join(self.comp_path, "test.csv")


class CompetitionData:
    """Represents data for a specific competition fold"""
    # TODO check if this approach is correct
    def __init__(self, train_path: os.PathLike, val_path: os.PathLike, fold_idx: int = 0,
                 additional_files: Dict[str, str] = None):
        self.train_path = train_path
        self.val_path = val_path
        self.fold_idx = fold_idx
        self.additional_files = additional_files or {}

    def get_train(self) -> os.PathLike:
        return self.train_path

    def get_val(self) -> os.PathLike:
        return self.val_path

    def get_additional_file(self, file_key: str) -> Optional[str]:
        """Get path to additional file"""
        return self.additional_files.get(file_key)

    def get_all_files(self) -> Dict[str, str]:
        """Get all files including train/val and additional files"""
        all_files = {
            "train": str(self.train_path),
            "val": str(self.val_path)
        }
        all_files.update(self.additional_files)
        return all_files
