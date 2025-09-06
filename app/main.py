"""
FastAPI application for fruit classification with Pydantic Settings.

This module provides an API endpoint for classifying fruit images
as fresh or rotten using a pre-trained ResNet model.
"""

import io
import logging
from typing import Dict, Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from PIL import Image
from pydantic import BaseModel
from torchvision.models import ResNet
from torchvision.transforms import v2 as transforms

from app.model import load_model, load_transforms, CATEGORIES
from app.config import get_settings, get_api_config

# Get configuration
settings = get_settings()
api_config = get_api_config()

# Setup logging
logger = logging.getLogger(__name__)


class PredictionResult(BaseModel):
    """
    Response model for prediction results.
    
    Attributes:
        category: The predicted fruit category
        confidence: Confidence score for the prediction (0.0 to 1.0)
    """
    category: str
    confidence: float


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    environment: str
    debug: bool


class CategoriesResponse(BaseModel):
    """Response model for categories endpoint."""
    categories: list[str]
    count: int


# Create the FastAPI instance with configuration
app = FastAPI(
    title=api_config.title,
    description=api_config.description,
    version=api_config.version,
    debug=settings.debug
)


def validate_file_size(file: UploadFile) -> None:
    """
    Validate uploaded file size.
    
    Args:
        file: Uploaded file to validate
        
    Raises:
        HTTPException: If file is too large
    """
    if hasattr(file, 'size') and file.size:
        if file.size > api_config.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {api_config.max_file_size / 1024 / 1024:.1f}MB"
            )


def validate_file_type(file: UploadFile) -> None:
    """
    Validate uploaded file type.
    
    Args:
        file: Uploaded file to validate
        
    Raises:
        HTTPException: If file type is not allowed
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image"
        )
    
    # Check file extension if filename is provided
    if file.filename:
        file_ext = '.' + file.filename.split('.')[-1].lower()
        if file_ext not in [ext.lower() for ext in api_config.allowed_extensions]:
            raise HTTPException(
                status_code=400,
                detail=f"File extension not allowed. Allowed: {', '.join(api_config.allowed_extensions)}"
            )


@app.get('/')
async def read_root() -> Dict[str, str]:
    """
    Root endpoint providing API information.
    
    Returns:
        Dict containing API information and available endpoints
    """
    return {
        'title': api_config.title,
        'version': api_config.version,
        'environment': settings.environment,
        'message': 'Use POST /predict with an image file for fruit classification',
        'endpoints': {
            'predict': 'POST /predict - Classify fruit image',
            'health': 'GET /health - Health check',
            'categories': 'GET /categories - Available categories'
        }
    }


@app.post('/predict', response_model=PredictionResult)
async def predict(
    input_image: UploadFile = File(..., description="Image file to classify"),
    model: ResNet = Depends(load_model),
    image_transforms: transforms.Compose = Depends(load_transforms)
) -> PredictionResult:
    """
    Predict fruit category from uploaded image.
    
    Args:
        input_image: Uploaded image file
        model: Pre-trained ResNet model (injected via dependency)
        image_transforms: Image preprocessing transforms (injected via dependency)
        
    Returns:
        PredictionResult with category and confidence score
        
    Raises:
        HTTPException: If image processing or prediction fails
    """
    try:
        # Validate file
        validate_file_size(input_image)
        validate_file_type(input_image)
        
        # Read and process image
        image_bytes = await input_image.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')

        # Apply transforms and add batch dimension
        processed_image = image_transforms(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = F.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)

        # Get category name
        category = CATEGORIES[predicted_class.item()]

        logger.info(f"Predicted {category} with confidence {confidence.item():.3f}")

        return PredictionResult(
            category=category, 
            confidence=confidence.item()
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ModelLoadError as e:
        # Обробляємо кастомний exception з model.py
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Model temporarily unavailable. Please try again later."
        ) from e
    except AppError as e:
        # Обробляємо загальні помилки додатка
        logger.error(f"Application error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal application error"
        ) from e
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        ) from e


@app.get('/health', response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint with configuration info.
    
    Returns:
        HealthResponse with service status and configuration
    """
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        debug=settings.debug
    )


@app.get('/categories', response_model=CategoriesResponse)
async def get_categories() -> CategoriesResponse:
    """
    Get available fruit categories.
    
    Returns:
        CategoriesResponse with list of available categories
    """
    return CategoriesResponse(
        categories=CATEGORIES,
        count=len(CATEGORIES)
    )


@app.get('/config')
async def get_config() -> Dict[str, Any]:
    """
    Get current API configuration (for debugging in development).
    
    Returns:
        Dict with current configuration
    """
    if settings.environment != 'development':
        raise HTTPException(
            status_code=404,
            detail="Config endpoint only available in development"
        )
    
    return {
        "environment": settings.environment,
        "debug": settings.debug,
        "api": {
            "title": api_config.title,
            "version": api_config.version,
            "max_file_size_mb": api_config.max_file_size / 1024 / 1024,
            "allowed_extensions": api_config.allowed_extensions
        },
        "categories": CATEGORIES
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {api_config.title} v{api_config.version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown."""
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.reload,
        log_level=settings.logging.level.lower()
    )