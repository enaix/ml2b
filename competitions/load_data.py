import os
import sys
import requests
import gdown
from pathlib import Path
import pandas as pd
import shutil

competition_map = {
    "widsdatathon2020": "wids-datathon-2020",
    "ieor242hw4": "ieor-242-nyc-taxi",
    "explicit-content-detection": "explicit-content-detection",
    "stat441datachallenge1": "uwaterloo-stat441-jewelry",
    "2020-10-29": "financial-engineering-1",
    "actuarial-loss-estimation": "actuarial-loss-prediction",
    "shaastra-wells-fargo-hackathon": "she-hacks-2021",
    "2021-ml-w1p1": "ai-cancer-predictions",
    "classification-with-non-deep-classifiers": "syde-522-winter-2021",
    "ventilator-pressure-prediction": "google-brain-ventilator",
    "porto-seguro-data-challenge": "porto-seguro-challenge",
    "crime-learn": "crime-learn",
    "playground-series-s3e2": "stroke-prediction-s3e2",
    "tabular-playground-series-aug-2021": "tabular-playground-series-aug-2021",
    "109-1-ntut-dl-app-hw1": "emnist-handwritten-chars",
    "made-hw-2": "movie-genre-classification",
    "prml-data-contest-nov-2020": "biker-tour-recommendation",
    "2020-11-05": "financial-engineering-2",
    "alfa-university-income-prediction": "alfa-university-income-prediction",
    "mlolympiadbd2025": "ml-olympiad-bd-2025",
    "playground-series-s5e6": "playground-series-s5e6",
    "ml2021spring-hw1": "ml2021spring-hw1",
    "2020-11-20": "financial-engineering-3",
    "thapar-summer-school-2025-hack-iii": "thapar-summer-school-2025-hack-iii",
    "rutgers-data101-fall2022-assignment-12": "rutgers-data101-fall2022-assignment-12",
    "2024-datalab-cup1": "2024-datalab-cup1",
    "ece460j-fall24": "ece460j-fall24",
    "playground-series-s5e3": "playground-series-s5e3",
    "cs-506-fall-2025-technical-midterm": "cs-506-fall-2025-technical-midterm",
    "car-becho-paisa-paao": "car-becho-paisa-paao",
    "itmo-flat-price-prediction-2024": "itmo-flat-price-prediction-2024",
    "2018-spring-cse6250-hw1": "2018-spring-cse6250-hw1",
    "ift-6390-ift-3395-beer-quality-prediction": "ift-6390-ift-3395-beer-quality-prediction",
    "multi-label-classification-competition-2025": "multi-label-classification-competition-2025",
}

DATA_URL = "https://drive.google.com/drive/folders/18QoNa3vjdJouI4bAW6wmGbJQCrWprxyf"
METADATA_URL = "https://docs.google.com/spreadsheets/d/1ZY8NRI-WZ4RoDK8GpEy_GTSSWVZPxQQthTaySp5jnao/export?format=csv&gid="

HF_DATASET = "enaix/ml2b"
HF_TASKS_DIR = "tasks"
HF_DATA_DIR = "data"


sheets = {
    "1525338984": "Arab.csv",
    "1607321930": "Belarus.csv",
    "940745352": "Chinese.csv",
    "658946950": "English.csv",
    "262900029": "Italian.csv",
    "1957726072": "Japanese.csv",
    "388903984": "Kazakh.csv",
    "2122623213": "Polish.csv",
    "1490673460": "Romanian.csv",
    "1065613449": "Spanish.csv",
    "0": "Turkish.csv",
    "751010443": "Russian.csv",
    "11714048": "French.csv",
}


def add_competition_id_to_all(directory_path: Path | str, competition_map: dict[str, str]) -> None:
    """
    Map competition id to competition metadata

    Args:
        directory_path (Path | str): path to metadata files
        competition_map (dict[str, str]): map competition id name to dataset compeition
    """
    folder = Path(directory_path)
    
    for csv_file in folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, sep=None, engine='python')
            df['comp-id'] = df['competition'].map(competition_map)
            
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"[OK] Added comp-id to {csv_file.name}")
            
        except Exception as e:
            print(f"[ERROR] Error with {csv_file.name}: {e}")


def load_data_gdrive() -> None:
    """
    Loads tasks data and metadata from GDrive, prepare for running benchmark
    """
    DEST = Path("competitions/tasks").resolve()
    DEST.mkdir(exist_ok=True)

    print(20 * "=" + f"Load competitions metadata to: {DEST}" + 20 * "=")
    for gid, filename in sheets.items():
        url = f"{METADATA_URL}{gid}"
        path = os.path.join(DEST, filename)
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        except requests.RequestException as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
    add_competition_id_to_all(DEST, competition_map)
    print(20 * "=" + "Load competitions data" + 20 * "=")
    gdown.download_folder(DATA_URL, output=str(DEST.parent / "data"), quiet=False, use_cookies=False)
    print("[OK] Benchmark data successfuly prepared")


def prompt_data_removal(path) -> None:
    yn = input(f"Folder {path} already exists. Remove? [y/N]: ")
    if yn.lower() not in ["y", "yes"]:
        print("Bailing out")
        sys.exit(0)


def load_data_huggingface(rm_cache: bool) -> None:
    """
    Loads tasks data and metadata from huggingface hub
    """
    #import datasets
    from huggingface_hub import snapshot_download

    TASKS = Path("competitions/tasks").resolve()
    if TASKS.exists():
        prompt_data_removal(TASKS)
        shutil.rmtree(TASKS)
    #TASKS.mkdir(exist_ok=True)

    DATA = Path("competitions/data").resolve()
    if DATA.exists():
        prompt_data_removal(DATA)
        shutil.rmtree(DATA)

    #DATA.mkdir(exist_ok=True)

    print("Downloading the dataset...")
    #tasks = datasets.load_dataset(HF_DATASET, data_dir=HF_TASKS_DIR)
    dataset = snapshot_download(repo_id=HF_DATASET, repo_type="dataset")

    #data = datasets.load_dataset(HF_DATASET, data_dir=HF_DATA_DIR)
    #data = snapshot_download(repo_id=HF_DATASET, subfolder=HF_DATA_DIR)

    tasks = (dataset / Path(HF_TASKS_DIR))
    data = (dataset / Path(HF_DATA_DIR))

    print("[OK] Path to the dataset cache:", dataset)
    if rm_cache:
        print("Moving task descriptions and data to the current folder...")
        shutil.move(tasks, TASKS)
        shutil.move(data, DATA)
    else:
        print("Copying task descriptions and data to the current folder...")
        shutil.copytree(tasks, TASKS)
        shutil.copytree(data, DATA)

    add_competition_id_to_all(TASKS, competition_map)
    print("[OK] Benchmark data successfuly prepared")


def print_help():
    print("Usage: load_data.py SOURCE REMOVE_CACHE")
    print("  SOURCE: [huggingface gdrive]")
    print("  REMOVE_CACHE: [remove_cache leave_cache]: to remove or leave huggingface cache to save space")
    sys.exit(1)


def load_data(source: str, rm_cache: bool):
    """
    Load the benchmark data from the source ("gdrive"/"huggingface")
    """

    if source == "gdrive":
        load_data_gdrive()
    elif source == "huggingface":
        load_data_huggingface(rm_cache)
    else:
        print("Bad data source:", source)
        print_help()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_help()

    remove_cache = sys.argv[2]
    if remove_cache == "remove_cache":
        rm_cache = True
    elif remove_cache == "leave_cache":
        rm_cache = False
    else:
        print("Bad cache option:", remove_cache)
        print_help()

    load_data(sys.argv[1], rm_cache)
