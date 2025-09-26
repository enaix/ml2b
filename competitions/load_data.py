import os
import requests
import gdown
from pathlib import Path
import pandas as pd

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
    
}

DATA_URL = "https://drive.google.com/drive/folders/18QoNa3vjdJouI4bAW6wmGbJQCrWprxyf"
METADATA_URL = "https://docs.google.com/spreadsheets/d/1ZY8NRI-WZ4RoDK8GpEy_GTSSWVZPxQQthTaySp5jnao/export?format=csv&gid="

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
            print(f"✅ Added comp-id to {csv_file.name}")
            
        except Exception as e:
            print(f"❌ Error with {csv_file.name}: {e}")


def load_data() -> None:
    """
    Loads tasks data and metadata, prepare for running benchmark
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
            print(f"❌ Failed to download {filename}: {e}")
    add_competition_id_to_all(DEST, competition_map)
    print(20 * "=" + "Load competitions data" + 20 * "=")
    gdown.download_folder(DATA_URL, output=str(DEST.parent / "data"), quiet=False, use_cookies=False)
    print("✅ Benchmark data successfuly prepared")

if __name__ == "__main__":
    load_data()