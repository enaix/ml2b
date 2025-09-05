from pathlib import Path
import re

def rename_files(directory_path):
    """Переименовывает файлы вида Arab_25.csv в Arab.csv"""
    folder = Path(directory_path)
    
    for file_path in folder.glob("*_*.csv"):
        new_name = re.sub(r'_\d+\.csv$', '.csv', file_path.name)
        
        if new_name != file_path.name:
            new_path = file_path.with_name(new_name)
            file_path.rename(new_path)
            print(f"Переименован: {file_path.name} -> {new_name}")

rename_files("./tasks")