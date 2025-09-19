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
    "ml2021spring-hw1": "ml2021spring-hw1"
}
import pandas as pd
from pathlib import Path

def add_competition_id_to_all(directory_path, competition_map):
    folder = Path(directory_path)
    
    for csv_file in folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, sep=None, engine='python')
            df['comp-id'] = df['competition'].map(competition_map)
            # Добавляем competition-id как первый столбец
            
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"✅ Добавлен comp-id в {csv_file.name}")
            
        except Exception as e:
            print(f"❌ Ошибка с {csv_file.name}: {e}")

# Использование

add_competition_id_to_all("tasks", competition_map)