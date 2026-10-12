from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"

RAW_HISTORICAL_FILE = ROOT_DIR / "MH_Crop_History_1997_2019.xlsx"
RAW_TRAINING_FILE = ROOT_DIR / "MH_Crop_Training_Data.xlsx"

PHASE_NAME = "phase_0"
