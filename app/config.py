"""
Application configuration using Pydantic Settings.

This module provides centralized configuration management with
automatic environment variable loading and validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseSettings, Field, validator


class WandBConfig(BaseSettings):
    """
    Weights & Biases configuration.
    
    All fields will be automatically populated from environment variables
    with WANDB_ prefix.
    """
    api_key: str = Field(..., description="W&B API key for authentication")
    org: str = Field(..., description="W&B organization name")
    project: str = Field(..., description="W&B project name")
    model_name: str = Field(..., description="W&B model artifact name")
    model_version: str = Field(default="latest", description="Model version to download")
    
    class Config:
        env_prefix = "WANDB_"  # Automatically maps WANDB_API_KEY to api_key
        case_sensitive = False


class ModelConfig(BaseSettings):
    """
    Model-specific configuration.
    """
    categories: List[str] = Field(
        default=[
            "freshapple", "freshbanana", "freshorange",
            "rottenapple", "rottenbanana", "rottenorange"
        ],
        description="Available fruit categories"
    )
    
    models_dir: Path = Field(default=Path("models"), description="Directory to store models")
    model_file_name: str = Field(default="best_model.pth", description="Model filename")
    expected_size_mb: float = Field(default=45.8, description="Expected model file size in MB")
    
    # Image preprocessing settings
    image_size: int = Field(default=224, description="Input image size")
    imagenet_mean: List[float] = Field(
        default=[0.485, 0.456, 0.406], 
        description="ImageNet normalization mean"
    )
    imagenet_std: List[float] = Field(
        default=[0.229, 0.224, 0.225], 
        description="ImageNet normalization std"
    )
    
    @validator('models_dir')
    def create_models_dir(cls, v):
        """Ensure models directory exists."""
        v.mkdir(exist_ok=True)
        return v
    
    class Config:
        env_prefix = "MODEL_"


class APIConfig(BaseSettings):
    """
    API server configuration.
    """
    title: str = Field(default="Fruit Classification API", description="API title")
    description: str = Field(
        default="API for classifying fruits as fresh or rotten",
        description="API description"
    )
    version: str = Field(default="1.0.0", description="API version")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload in development")
    
    # File upload settings
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max upload size in bytes")  # 10MB
    allowed_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"],
        description="Allowed file extensions"
    )
    
    class Config:
        env_prefix = "API_"


class LoggingConfig(BaseSettings):
    """
    Logging configuration.
    """
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    class Config:
        env_prefix = "LOG_"


class AppSettings(BaseSettings):
    """
    Main application settings that combines all configuration sections.
    """
    # Environment
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Configuration sections
    wandb: WandBConfig = WandBConfig()
    model: ModelConfig = ModelConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    
    @validator('environment')
    def validate_environment(cls, v):
        """Ensure environment is valid."""
        allowed_envs = ['development', 'staging', 'production']
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v
    
    class Config:
        # Load from .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        # Allow nested environment variables like WANDB__API_KEY
        env_nested_delimiter = "__"
        
        # Case insensitive environment variables
        case_sensitive = False


@lru_cache()
def get_settings() -> AppSettings:
    """
    Get cached application settings.
    
    Using lru_cache ensures settings are loaded only once
    and reused across the application.
    
    Returns:
        AppSettings instance with all configuration
    """
    return AppSettings()


# Convenience functions to get specific config sections
def get_wandb_config() -> WandBConfig:
    """Get W&B configuration."""
    return get_settings().wandb


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return get_settings().model


def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_settings().api


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_settings().logging


# Example usage and validation
if __name__ == "__main__":
    # Load settings
    settings = get_settings()
    
    print("=== Application Settings ===")
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    
    print("\n=== W&B Config ===")
    try:
        print(f"Organization: {settings.wandb.org}")
        print(f"Project: {settings.wandb.project}")
        print(f"Model: {settings.wandb.model_name}:{settings.wandb.model_version}")
    except Exception as e:
        print(f"W&B config error (check your .env file): {e}")
    
    print("\n=== Model Config ===")
    print(f"Models directory: {settings.model.models_dir}")
    print(f"Categories: {settings.model.categories}")
    print(f"Image size: {settings.model.image_size}")
    
    print("\n=== API Config ===")
    print(f"Title: {settings.api.title}")
    print(f"Server: {settings.api.host}:{settings.api.port}")
    print(f"Max file size: {settings.api.max_file_size / 1024 / 1024:.1f}MB")
    
    # Export to JSON for documentation
    print("\n=== Full Configuration (JSON) ===")
    try:
        print(settings.json(indent=2))
    except Exception as e:
        print(f"Note: Some values may be missing from .env file: {e}")