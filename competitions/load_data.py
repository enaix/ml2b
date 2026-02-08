import sys
from pathlib import Path
import pandas as pd
import shutil
from huggingface_hub import snapshot_download

competition_map = {
    "widsdatathon2020": "wids-datathon-2020",
    "ieor242hw4": "ieor-242-nyc-taxi",
    "stat441datachallenge1": "uwaterloo-stat441-jewelry",
    "2020-10-29": "financial-engineering-1",
    "shaastra-wells-fargo-hackathon": "she-hacks-2021",
    "2021-ml-w1p1": "ai-cancer-predictions",
    "classification-with-non-deep-classifiers": "syde-522-winter-2021",
    "ventilator-pressure-prediction": "google-brain-ventilator",
    "porto-seguro-data-challenge": "porto-seguro-challenge",
    "playground-series-s3e2": "stroke-prediction-s3e2",
    "109-1-ntut-dl-app-hw1": "emnist-handwritten-chars",
    "made-hw-2": "movie-genre-classification",
    "prml-data-contest-nov-2020": "biker-tour-recommendation",
    "2020-11-05": "financial-engineering-2",
    "mlolympiadbd2025": "ml-olympiad-bd-2025",
    "2020-11-20": "financial-engineering-3",
}


def add_competition_id_to_all(
    directory_path: Path | str, competition_map: dict[str, str]
) -> None:
    """Map competition id to competition metadata

    Args:
        directory_path (Path | str): path to metadata files
        competition_map (dict[str, str]): map competition id name to dataset compeition
    """
    folder = Path(directory_path)

    for csv_file in folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, sep=None, engine="python")
            df["comp-id"] = (
                df["competition"].map(competition_map).fillna(df["competition"])
            )

            df.to_csv(csv_file, index=False, encoding="utf-8")
            print(f"[OK] Added comp-id to {csv_file.name}")

        except Exception as e:
            print(f"[ERROR] Error with {csv_file.name}: {e}")


def prompt_data_removal(path: Path | str) -> None:
    """Helper func"""
    yn = input(f"Folder {path} already exists. Remove? [y/N]: ")
    if yn.lower() not in ["y", "yes"]:
        print("Bailing out")
        sys.exit(0)


def load_data_huggingface(
    data_dir: Path | str,
    rm_cache: bool,
    hf_dataset: str,
    hf_tasks_dir: str,
    hf_data_dir: str,
) -> None:
    """Loads tasks data and metadata from huggingface hub

    Args:
        data_dir (PathLike): Path to store competition data
        rm_cache (bool): Remove cache after loading
        hf_dataset (str): Huggingface dataset name
        hf_tasks_dir (str): Tasks dir in hf store
        hf_data_dir (str): Data dir in hf store
    """
    data_dir = Path(data_dir).resolve()
    tasks_dir = data_dir / "tasks"
    if tasks_dir.exists():
        prompt_data_removal(tasks_dir)
        shutil.rmtree(tasks_dir)

    dataset_dir = data_dir / "data"
    if dataset_dir.exists():
        prompt_data_removal(dataset_dir)
        shutil.rmtree(dataset_dir)

    print("Downloading the dataset...")
    dataset = Path(snapshot_download(repo_id=hf_dataset, repo_type="dataset"))

    tasks = dataset / hf_tasks_dir
    data = dataset / hf_data_dir

    print("[OK] Path to the dataset cache:", dataset)
    if rm_cache:
        print("Moving task descriptions and data to the current folder...")
        shutil.move(tasks, tasks_dir)
        shutil.move(data, dataset_dir)
    else:
        print("Copying task descriptions and data to the current folder...")
        shutil.copytree(tasks, tasks_dir)
        shutil.copytree(data, dataset_dir)

    add_competition_id_to_all(tasks_dir, competition_map)
    print("[OK] Benchmark data successfuly prepared")


def load_data(
    data_dir: Path | str,
    hf_dataset: str,
    hf_tasks_dir: str,
    hf_data_dir: str,
    source: str = "huggingface",
    rm_cache: bool = False,
) -> None:
    """Load benchmark data from source

    Args:
        data_dir (PathLike): Path to store competition data
        hf_dataset (str): Huggingface dataset name
        hf_tasks_dir (str): Tasks dir in hf store
        hf_data_dir (str): Data dir in hf store
        source (str, optional): Data source. Defaults to "huggingface".
        rm_cache (bool, optional): Remove cache after loading. Defaults to False.
    """
    if data_dir is None:
        print("Provide a path to store competition data")
        return
    if source == "huggingface":
        load_data_huggingface(data_dir, rm_cache, hf_dataset, hf_tasks_dir, hf_data_dir)
    else:
        print("Bad data source:", source)
