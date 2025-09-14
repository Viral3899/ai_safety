# Configuration settings for ai_safety_poc

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = DATA_DIR / 'models'

# Example runtime/config settings
DEBUG = True
DEFAULT_THRESHOLD = 0.5
