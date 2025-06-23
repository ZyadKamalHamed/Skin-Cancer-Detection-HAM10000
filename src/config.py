"""
Configuration management for Skin Cancer Detection System.

This module centralizes all configuration parameters for reproducible experiments
and easy hyperparameter tuning.
"""

import os
from pathlib import Path
from typing import Dict

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model subdirectories
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
BEST_MODELS_DIR = MODELS_DIR / "best_models"
ENSEMBLE_DIR = MODELS_DIR / "ensemble"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR,
                  RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  CHECKPOINTS_DIR, BEST_MODELS_DIR, ENSEMBLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# HAM10000 Dataset Configuration
DATASET_CONFIG = {
    "name": "HAM10000",
    "num_classes": 7,
    "class_names": [
        "Melanocytic nevi",
        "Melanoma",
        "Benign keratosis-like lesions",
        "Basal cell carcinoma",
        "Actinic keratoses",
        "Vascular lesions",
        "Dermatofibroma"
    ],
    "class_labels": ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"],
    "image_size": (224, 224),
    "channels": 3,
    "download_urls": {
        "images_part1": "https://dataverse.harvard.edu/api/access/datafile/3450625",
        "images_part2": "https://dataverse.harvard.edu/api/access/datafile/3450626",
        "metadata": "https://dataverse.harvard.edu/api/access/datafile/3450624"
    }
}

# Class distribution (for handling imbalanced dataset)
CLASS_DISTRIBUTION = {
    "nv": 6705,  # Melanocytic nevi (most common)
    "mel": 1113,  # Melanoma
    "bkl": 1099,  # Benign keratosis-like lesions
    "bcc": 514,  # Basal cell carcinoma
    "akiec": 327,  # Actinic keratoses
    "vasc": 142,  # Vascular lesions
    "df": 115  # Dermatofibroma (least common)
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    "resnet50": {
        "name": "ResNet50",
        "pretrained": True,
        "num_classes": 7,
        "feature_dim": 2048,
        "dropout_rate": 0.5,
        "freeze_backbone": False  # Allow fine-tuning
    },
    "densenet121": {
        "name": "DenseNet121",
        "pretrained": True,
        "num_classes": 7,
        "feature_dim": 1024,
        "dropout_rate": 0.5,
        "freeze_backbone": False
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "pretrained": True,
        "num_classes": 7,
        "feature_dim": 1280,
        "dropout_rate": 0.5,
        "freeze_backbone": False
    }
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "momentum": 0.9,


    # Learning rate scheduling
    "scheduler": "cosine",  # Options: "step", "cosine", "plateau"
    "step_size": 10,
    "gamma": 0.1,
    "patience": 5,

    # Early stopping
    "early_stopping_patience": 10,
    "early_stopping_delta": 0.001,

    # Mixed precision training
    "use_amp": True,

    # Gradient clipping
    "gradient_clip_val": 1.0,

    # Validation frequency
    "val_check_interval": 1,

    # Checkpointing
    "save_top_k": 3,
    "monitor": "val_auc",
    "mode": "max"
}

# =============================================================================
# DATA AUGMENTATION CONFIGURATION
# =============================================================================

AUGMENTATION_CONFIG = {
    "train": {
        "horizontal_flip": {"p": 0.5},
        "vertical_flip": {"p": 0.5},
        "rotation": {"limit": 45, "p": 0.5},
        "random_brightness_contrast": {
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "p": 0.5
        },
        "hue_saturation_value": {
            "hue_shift_limit": 20,
            "sat_shift_limit": 30,
            "val_shift_limit": 20,
            "p": 0.5
        },
        "gaussian_blur": {"blur_limit": 3, "p": 0.3},
        "elastic_transform": {
            "alpha": 1,
            "sigma": 50,
            "alpha_affine": 50,
            "p": 0.3
        },
        "normalize": {
            "mean": [0.485, 0.456, 0.406],  # ImageNet mean
            "std": [0.229, 0.224, 0.225]  # ImageNet std
        }
    },
    "val": {
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    },
    "test": {
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVALUATION_CONFIG = {
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc_roc",
        "auc_pr",
        "specificity",
        "sensitivity"
    ],
    "average": "macro",  # For multi-class metrics
    "bootstrap_samples": 1000,  # For confidence intervals
    "confidence_level": 0.95,
    "class_wise_metrics": True,
    "confusion_matrix": True,
    "roc_curve": True,
    "precision_recall_curve": True
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

VISUALIZATION_CONFIG = {
    "grad_cam": {
        "target_layers": ["layer4"],  # For ResNet
        "use_cuda": True,
        "aug_smooth": True,
        "eigen_smooth": True
    },
    "plot_style": "seaborn-v0_8-darkgrid",
    "figure_size": (12, 8),
    "dpi": 300,
    "save_format": "png",
    "color_palette": "husl"
}

# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================

DEPLOYMENT_CONFIG = {
    "streamlit": {
        "title": "🔬 Skin Cancer Detection System",
        "max_file_size": 10,  # MB
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "confidence_threshold": 0.7,
        "show_gradcam": True,
        "show_probabilities": True
    },
    "huggingface": {
        "model_name": "skin-cancer-detection",
        "readme_template": "deployment/hf_readme.md",
        "examples": ["example1.jpg", "example2.jpg", "example3.jpg"]
    }
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "log_file": LOGS_DIR / "training.log",
    "wandb": {
        "project": "skin-cancer-detection",
        "entity": "your-wandb-username",  # Update with your W&B username
        "tags": ["skin-cancer", "medical-ai", "computer-vision"]
    }
}

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================

HARDWARE_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "num_workers": min(8, os.cpu_count()),
    "pin_memory": True,
    "persistent_workers": True
}

# =============================================================================
# REPRODUCIBILITY CONFIGURATION
# =============================================================================

REPRODUCIBILITY_CONFIG = {
    "seed": 42,
    "deterministic": True,
    "benchmark": False  # Set to True for consistent input sizes
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_class_weights() -> Dict[str, float]:
    """Calculate class weights for handling imbalanced dataset."""
    total_samples = sum(CLASS_DISTRIBUTION.values())
    num_classes = len(CLASS_DISTRIBUTION)

    class_weights = {}
    for class_name, count in CLASS_DISTRIBUTION.items():
        weight = total_samples / (num_classes * count)
        class_weights[class_name] = weight

    return class_weights


def get_model_config(model_name: str) -> Dict:
    """Get configuration for specific model."""
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Model {model_name} not found in configuration")
    return MODEL_CONFIG[model_name]


def get_device() -> str:
    """Get the appropriate device for training/inference."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():  # For Apple Silicon
        return "mps"
    else:
        return "cpu"


# Update hardware config with actual device
HARDWARE_CONFIG["device"] = get_device()

# Class weights for loss function
CLASS_WEIGHTS = get_class_weights()

# Export key configurations
__all__ = [
    "DATASET_CONFIG",
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
    "AUGMENTATION_CONFIG",
    "EVALUATION_CONFIG",
    "VISUALIZATION_CONFIG",
    "DEPLOYMENT_CONFIG",
    "LOGGING_CONFIG",
    "HARDWARE_CONFIG",
    "REPRODUCIBILITY_CONFIG",
    "CLASS_WEIGHTS",
    "ROOT_DIR",
    "DATA_DIR",
    "MODELS_DIR"
]