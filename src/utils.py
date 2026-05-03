from pathlib import Path

def ensure_directories():
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
