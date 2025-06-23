"""Configuration for Skin Cancer Detection System"""
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Dataset config
DATASET_CONFIG = {
    "num_classes": 7,
    "class_names": ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"],
    "image_size": (224, 224),
    "batch_size": 16,  # Added this line that was missing
    "channels": 3
}

# Training config
TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "epochs": 30,
    "device": "cpu"
}

print(" Config loaded successfully!")
print(f" Data directory: {DATA_DIR}")
print(f" Models directory: {MODELS_DIR}")
print(f" Number of classes: {DATASET_CONFIG['num_classes']}")