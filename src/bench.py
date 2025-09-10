import os
import sys
from enum import Enum, StrEnum
import weakref
import traceback
import json
import subprocess
from pathlib import Path
import shutil
import docker
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
import importlib
from docker import DockerClient
from loguru import logger

class Language(StrEnum):
    English = "English"
    Arab = "Arab"
    Chinese = "Chinese"
    Italian = "Italian"
    Kazakh = "Kazakh"
    Polish = "Polish"
    Romanian = "Romanian"
    Spanish = "Spanish"
    Turkish = "Turkish"
    Belarus = "Belarus"
    Japanese = "Japanese"


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
    def __init__(self, comp_id: str, bench: weakref.ReferenceType, metadata: dict, tasks: dict, 
                 grading_stage: bool = False):
        """
        Represents a machine learning competition.
        
        Args:
            comp_id: Unique competition identifier
            bench: Weak reference to the benchmark pipeline
            metadata: Competition configuration and metadata
            tasks: Task descriptions for different languages
            grading_stage: If True, skip file validation (for Docker grading stage)
        """
        self.comp_id = comp_id
        self.metadata = metadata
        self.bench = bench
        self.tasks = tasks
        self.grading_stage = grading_stage
        self.files = None
        
        # Set competition path based on stage
        if grading_stage:
            self.comp_path = os.path.join('/bench/data/', "data", comp_id)
        else:
            self.comp_path = os.path.join(bench().base_path(), "competitions", "data", comp_id)
        
        # Initialize files based on metadata or discovery
        # self.files = self._initialize_files()
        
        # Validate files unless we're in grading stage
        # if not grading_stage:
        #     self._validate_files()

    def initialize_files(self, files) -> None:
        """Set competition files"""
        self.files = files


    def _initialize_files(self) -> Dict[str, CompetitionFile]:
        """Initialize competition files from metadata or by discovery"""
        files = {}
        
        # Get file configuration from metadata
        file_config = self.metadata.get("files", {
            "train": {"type": "data", "required": True, "extensions": [".csv"]},
        })
        
        # First, try to use explicit file mapping from metadata
        explicit_files = self.metadata.get("file_mapping", {})
        for file_key, file_info in explicit_files.items():
            file_path = os.path.join(self.comp_path, file_info["filename"])
            files[file_key] = CompetitionFile(
                name=file_key,
                path=file_path,
                file_type=file_info.get("type", "data"),
                required=file_info.get("required", True)
            )
        
        # If no explicit mapping, discover files based on configuration
        if not explicit_files:
            for file_key, config in file_config.items():
                file_found = False
                for ext in config.get("extensions", [".csv"]):
                    potential_path = os.path.join(self.comp_path, f"{file_key}{ext}")
                    
                    # Only check existence if we're not in grading stage
                    if not self.grading_stage:
                        file_exists = os.path.exists(potential_path)
                    else:
                        file_exists = True
                    
                    if file_exists:
                        files[file_key] = CompetitionFile(
                            name=file_key,
                            path=potential_path,
                            file_type=config.get("type", "data"),
                            required=config.get("required", True)
                        )
                        file_found = True
                        break
                
                # If required file not found and we're not in grading stage, create entry for validation
                if not file_found and config.get("required", True) and not self.grading_stage:
                    potential_path = os.path.join(self.comp_path, f"{file_key}.csv")
                    files[file_key] = CompetitionFile(
                        name=file_key,
                        path=potential_path,
                        file_type=config.get("type", "data"),
                        required=True
                    )
        
        # Discover additional files
        if os.path.exists(self.comp_path) or self.grading_stage:
            for item in os.listdir(self.comp_path):
                item_path = os.path.join(self.comp_path, item)
                
                # Skip if we're not in grading stage and path doesn't exist
                if not self.grading_stage and not os.path.exists(item_path):
                    continue
                
                if os.path.isfile(item_path):
                    filename = item
                    if self._is_submission_file(filename):
                        continue
                    
                    file_key = os.path.splitext(filename)[0]
                    if file_key not in files:
                        file_type = self._infer_file_type(filename)
                        if file_type in ["data", "metadata"]:
                            files[file_key] = CompetitionFile(
                                name=file_key,
                                path=item_path,
                                file_type=file_type,
                                required=False
                            )
                
                elif os.path.isdir(item_path):
                    dir_name = item
                    if not self._is_submission_dir(dir_name):
                        files[dir_name] = CompetitionFile(
                            name=dir_name,
                            path=item_path,
                            file_type="data",
                            required=False
                        )
        
        return files
    
    def _is_submission_dir(self, dirname: str) -> bool:
        """Check if a directory is submission-related"""
        dirname_lower = dirname.lower()
        submission_keywords = ["submission", "sample", "baseline", "example"]
        return any(keyword in dirname_lower for keyword in submission_keywords)

    def _is_submission_file(self, filename: str) -> bool:
        """Check if a file is submission-related"""
        filename_lower = filename.lower()
        submission_keywords = [
            "sample_submission", "samplesubmission", "submission", 
            "submit", "example_submission", "baseline"
        ]
        return any(keyword in filename_lower for keyword in submission_keywords)
    
    def _infer_file_type(self, filename: str) -> str:
        """Infer file type from filename"""
        filename_lower = filename.lower()
        
        if filename_lower.endswith(('.csv', '.json', '.parquet', '.pkl', '.pickle', '.h5', '.hdf5')):
            return "data"
        elif filename_lower.endswith(('.txt', '.md', '.json', '.xml', '.yml', '.yaml')):
            if any(word in filename_lower for word in ["description", "readme", "info", "meta"]):
                return "metadata"
            else:
                return "data"
        elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif')):
            return "data"
        elif filename_lower.endswith(('.npy', '.npz', '.mat')):
            return "data"
        else:
            return "other"
    
    def _validate_files(self) -> None:
        """Validate that all required files exist"""
        missing_files = []
        for file_key, comp_file in self.files.items():
            if comp_file.required and not comp_file.exists():
                missing_files.append(f"{file_key} ({comp_file.path})")
        
        if missing_files:
            error_msg = f"Competition {self.comp_id}: missing required files: {', '.join(missing_files)}"
            if self.bench() is not None:
                self.bench().shutdown(1)
            else:
                raise FileNotFoundError(error_msg)
    
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

            self.competitions.append(Competition(key, weakref.ref(self), value, comp_tasks))
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

    def test_submission_code(self, comp: Competition, lang: Language, codelang: CodeLanguage, code: str) -> dict:
        """
        Submit the code and return metric
        """

        # TODO refactor this

        # Prepare submission dir
        # ======================
        submission_dir = (Path(self.base_path()) / str(codelang) / f"submission_{uniq_suf}").resolve()
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
            "BENCH_LANG": str(lang),
            "BENCH_MODE": str(BenchMode.ModularPredict),
            "BENCH_FOLDS_OVERRIDE": "1",
            "PYTHONDONTWRITEBYTECODE": "1"
        }
        network_name = "python_no_inet"

        # Проверяем, есть ли сеть
        networks = [n.name for n in client.networks.list()]
        if network_name not in networks:
            client.networks.create(network_name, driver="bridge", internal=True)
        container = client.containers.run(
            image=image_name,
            detach=True,
            environment=env_vars,
            volumes={
                submission_dir.as_posix(): {'bind': '/home/bench/submission', 'mode': 'rw'},
                (Path(self.base_path()) / "competitions").resolve().as_posix(): {'bind': '/home/bench/competitions', 'mode': 'ro'}
            },
            network="python_no_inet",
            entrypoint=["mamba", "run", "-n", "agent", "python", "./bench.py"],
            **runtime_config,
            working_dir="/home/bench"
        )

        exit_code = container.wait(timeout=60*60)
        logs = container.logs().decode('utf-8')

        logger.info("Evaluation container results:\n exit_code: {}\n logs: {}", exit_code, logs)
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

    def prepare_train_data(self, comp: Competition) -> None:
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

        # Get splitting strategy
        split_strategy = comp.metadata.get("split_strategy", "csv")
        splitter_class = DATA_SPLITTERS.get(split_strategy)
        
        if not splitter_class:
            print(f"Unknown split strategy '{split_strategy}' for competition {comp.comp_id}")
            print(f"Available strategies: {list(DATA_SPLITTERS.keys())}")
            self.shutdown(1)
        
        try:
            splitter = splitter_class()
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


    # TODO remove dead code

    def register_custom_splitter(self, name: str, splitter_class: type):
        """Register a custom data splitter"""
        if not issubclass(splitter_class, DataSplitter):
            raise ValueError("Custom splitter must inherit from DataSplitter")
        DATA_SPLITTERS[name] = splitter_class

    def get_available_splitters(self) -> List[str]:
        """Get available splitting strategies"""
        return list(DATA_SPLITTERS.keys())

    def register_custom_loader(self, name: str, loader_class: type):
        """Register a custom data loader"""
        if not issubclass(loader_class, DataLoader):
            raise ValueError("Custom loader must inherit from DataLoader")
        DATA_LOADERS[name] = loader_class

    def get_available_loaders(self) -> List[str]:
        """Get available data loaders"""
        return list(DATA_LOADERS.keys())
