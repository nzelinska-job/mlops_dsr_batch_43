"""
Fruit Classification API Package.

This package provides a FastAPI-based web service for classifying
fruit images as fresh or rotten using a pre-trained ResNet model.

Usage:
    from app import app, get_settings
    
    # Run the app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

Modules:
    main: FastAPI application with prediction endpoints
    model: Model loading and preprocessing utilities  
    config: Configuration management with Pydantic Settings
"""

import logging
from pathlib import Path

__version__ = "1.0.0"
__author__ = "Nataliia Zelinska"
__description__ = "Fruit Classification API using PyTorch and FastAPI"

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Package-level logger
logger = logging.getLogger(__name__)

# Public API - імпортуємо тільки те, що потрібно зовнішнім користувачам
from .main import app
from .model import CATEGORIES, load_model, load_transforms
from .config import get_settings

# Exception classes для кращої обробки помилок
class AppError(Exception):
    """Base exception for application errors."""
    pass

class ModelLoadError(AppError):
    """Raised when model loading fails."""
    pass

class ConfigurationError(AppError):  
    """Raised when configuration is invalid."""
    pass

# Експортуємо публічний API
__all__ = [
    # Main application
    "app",
    
    # Model utilities
    "CATEGORIES", 
    "load_model", 
    "load_transforms",
    
    # Configuration
    "get_settings",
    
    # Exceptions
    "AppError",
    "ModelLoadError", 
    "ConfigurationError",
    
    # Metadata
    "__version__",
]

# Валідація пакета при імпорті (опціонально)
def _validate_package():
    """Validate package requirements on import."""
    try:
        # Перевіряємо критичні залежності
        import torch
        import fastapi
        import pydantic
        logger.debug("Package dependencies validated successfully")
    except ImportError as e:
        logger.error(f"Missing critical dependency: {e}")
        raise ImportError(f"Failed to import required dependency: {e}")

# Запускаємо валідацію
_validate_package()

# Package-level константи, доступні всім модулям
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Utility function для зручного запуску
def run_app(host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """
    Convenience function to run the FastAPI application.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        **kwargs: Additional arguments passed to uvicorn.run()
    """
    try:
        import uvicorn
        settings = get_settings()
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=settings.logging.level.lower(),
            **kwargs
        )
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        raise
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise AppError(f"Application startup failed: {e}") from e

# Health check для пакета
def health_check():
    """
    Perform basic health check of the package.
    
    Returns:
        dict: Health status information
    """
    try:
        settings = get_settings()
        return {
            "status": "healthy",
            "version": __version__,
            "environment": settings.environment,
            "debug": settings.debug,
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "version": __version__
        }