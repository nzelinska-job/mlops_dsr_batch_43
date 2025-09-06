"""
Model loading and configuration module.

This module handles downloading pre-trained models from Weights & Biases,
model architecture setup, and image preprocessing transforms.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import wandb
from torch import nn
from torchvision.models import resnet18, ResNet
from torchvision.transforms import v2 as transforms

# NOTE: Remove these imports for GCP deployment
try:
    from dotenv import load_dotenv 
    load_dotenv()  
except ImportError:
    logging.warning("python-dotenv not available, skipping .env file loading")


# Configuration constants
CATEGORIES = [
    "freshapple", "freshbanana", "freshorange",
    "rottenapple", "rottenbanana", "rottenorange"
]

MODELS_DIR = Path('models')
MODEL_FILE_NAME = 'best_model.pth'
MODEL_EXPECTED_SIZE_MB = 45.8  # Expected model file size for validation

# Image preprocessing constants
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Create models directory
MODELS_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """
    Validate that required environment variables are set.
    
    Raises:
        EnvironmentError: If required environment variables are missing
    """
    required_vars = [
        'WANDB_API_KEY', 'WANDB_ORG', 'WANDB_PROJECT', 
        'WANDB_MODEL_NAME', 'WANDB_MODEL_VERSION'
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def download_artifact() -> None:
    """
    Download model artifact from Weights & Biases.
    
    Raises:
        EnvironmentError: If required environment variables are missing
        wandb.errors.CommError: If download fails
    """
    validate_environment()
    
    wandb_org = os.environ['WANDB_ORG']
    wandb_project = os.environ['WANDB_PROJECT']
    wandb_model_name = os.environ['WANDB_MODEL_NAME']
    wandb_model_version = os.environ['WANDB_MODEL_VERSION']

    artifact_path = f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"

    try:
        wandb.login()
        logger.info(f"Downloading artifact {artifact_path} to {MODELS_DIR}")
        
        api = wandb.Api()
        artifact = api.artifact(artifact_path, type='model')
        artifact.download(root=str(MODELS_DIR))
        
        logger.info("Model artifact downloaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to download artifact: {e}")
        raise


def create_model_architecture() -> ResNet:
    """
    Create ResNet18 model with custom classifier head.
    
    Returns:
        ResNet model with modified final layer for fruit classification
    """
    model = resnet18(weights=None)
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),  # Add dropout for better generalization
        nn.Linear(512, len(CATEGORIES))
    )
    
    return model


def validate_model_file(model_path: Path) -> None:
    """
    Validate that the model file exists and has expected properties.
    
    Args:
        model_path: Path to the model file
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file size is unexpected
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    expected_size = MODEL_EXPECTED_SIZE_MB
    
    # Allow for some variance in file size
    if not (expected_size * 0.9 <= file_size_mb <= expected_size * 1.1):
        logger.warning(
            f"Model file size ({file_size_mb:.1f} MB) differs from expected "
            f"size ({expected_size} MB). This may indicate a corrupted download."
        )


def load_model() -> ResNet:
    """
    Load pre-trained model with weights from Weights & Biases.
    
    Returns:
        Trained ResNet model ready for inference
        
    Raises:
        FileNotFoundError: If model file is not found
        RuntimeError: If model loading fails
    """
    model_path = MODELS_DIR / MODEL_FILE_NAME
    
    # Download model if not present locally
    if not model_path.exists():
        logger.info("Model not found locally, downloading from W&B")
        download_artifact()
    
    # Validate model file
    validate_model_file(model_path)
    
    try:
        # Create model architecture
        model = create_model_architecture()
        
        # Load trained weights
        logger.info(f"Loading model weights from {model_path}")
        state_dict = torch.load(
            model_path, 
            map_location='cpu',
            weights_only=True  # Security best practice
        )
        
        # Load weights with strict checking
        model.load_state_dict(state_dict, strict=True)
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e


def load_transforms() -> transforms.Compose:
    """
    Create image preprocessing transforms for model input.
    
    Returns:
        Composed transforms for image preprocessing
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


# Cache model instance for reuse (optional optimization)
_model_cache: Optional[ResNet] = None


def get_cached_model() -> ResNet:
    """
    Get cached model instance or load if not cached.
    
    Returns:
        Cached model instance
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model()
    return _model_cache


# Initialize model on module import (for development/testing)
if __name__ == "__main__":
    logger.info("Loading model for testing...")
    test_model = load_model()
    logger.info(f"Model architecture:\n{test_model}")
    logger.info(f"Available categories: {CATEGORIES}")