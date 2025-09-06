# =============================================================================
# ПРИКЛАДИ ВИКОРИСТАННЯ НОВОГО __init__.py API
# =============================================================================

# =============================================================================
# 1. Запуск FastAPI додатка (основний use case)
# =============================================================================

# run_server.py
from app import app, run_app, get_settings

# Варіант 1: Через uvicorn напряму
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        app, 
        host=settings.api.host, 
        port=settings.api.port
    )

# Варіант 2: Через зручну функцію з __init__.py
if __name__ == "__main__":
    run_app(host="0.0.0.0", port=8080, reload=True)

# =============================================================================
# 2. Тестування (tests/test_app.py)
# =============================================================================

import pytest
from fastapi.testclient import TestClient
from app import app, CATEGORIES, health_check, AppError, ModelLoadError

def test_health_check():
    """Test package health check function."""
    health = health_check()
    assert health["status"] in ["healthy", "unhealthy"]
    assert "version" in health

def test_categories():
    """Test categories are available."""
    assert isinstance(CATEGORIES, list)
    assert len(CATEGORIES) > 0
    assert "freshapple" in CATEGORIES

def test_api_endpoints():
    """Test FastAPI endpoints."""
    client = TestClient(app)
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    
    # Test categories endpoint
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert data["categories"] == CATEGORIES

def test_custom_exceptions():
    """Test custom exceptions are available."""
    with pytest.raises(AppError):
        raise AppError("Test error")
    
    with pytest.raises(ModelLoadError):
        raise ModelLoadError("Model loading failed")

# =============================================================================
# 3. Debugging та моніторинг (debug_app.py)
# =============================================================================

from app import health_check, get_settings, __version__
import json

def debug_info():
    """Print debugging information about the application."""
    print(f"=== App Debug Info ===")
    print(f"Version: {__version__}")
    
    # Health check
    health = health_check()
    print(f"Health: {json.dumps(health, indent=2)}")
    
    # Settings
    settings = get_settings()
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    print(f"Categories: {', '.join(CATEGORIES)}")

if __name__ == "__main__":
    debug_info()

# =============================================================================
# 4. Використання як бібліотеки (external_usage.py)  
# =============================================================================

# Зовнішній код, який використовує ваш пакет
from app import load_model, load_transforms, CATEGORIES, ModelLoadError
from PIL import Image
import torch

def classify_image(image_path: str):
    """
    Classify single image using the trained model.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Prediction result
    """
    try:
        # Load model and transforms
        model = load_model()
        transforms = load_transforms()
        
        # Load and preprocess image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        processed_image = transforms(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
        
        category = CATEGORIES[predicted_class.item()]
        
        return {
            'category': category,
            'confidence': confidence.item(),
            'all_categories': CATEGORIES
        }
        
    except ModelLoadError as e:
        print(f"Failed to load model: {e}")
        return None
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

# Використання
if __name__ == "__main__":
    result = classify_image("path/to/your/fruit.jpg")
    if result:
        print(f"Prediction: {result['category']} ({result['confidence']:.3f})")

# =============================================================================
# 5. Docker entry point (docker_entrypoint.py)
# =============================================================================

#!/usr/bin/env python3
"""Docker container entry point."""

import sys
import logging
from app import run_app, health_check, get_settings, ConfigurationError

def main():
    """Main entry point for Docker container."""
    try:
        # Check application health
        health = health_check()
        if health["status"] != "healthy":
            logging.error(f"Application health check failed: {health}")
            sys.exit(1)
        
        # Get settings
        settings = get_settings()
        
        # Run application
        logging.info(f"Starting application in {settings.environment} mode")
        run_app(
            host="0.0.0.0",
            port=8000,
            log_level=settings.logging.level.lower()
        )
        
    except ConfigurationError as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# =============================================================================
# 6. CLI інтерфейс (cli.py)
# =============================================================================

import argparse
from pathlib import Path
from app import CATEGORIES, health_check, __version__, run_app

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Fruit Classification API v{__version__}"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start the API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    server_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check application health')
    
    # Categories command
    categories_parser = subparsers.add_parser('categories', help='List available categories')
    
    return parser

def main():
    """CLI main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == 'server':
        print(f"Starting Fruit Classification API v{__version__}")
        run_app(host=args.host, port=args.port, reload=args.reload)
        
    elif args.command == 'health':
        health = health_check()
        print(f"Status: {health['status']}")
        if health['status'] != 'healthy':
            print(f"Error: {health.get('error', 'Unknown error')}")
            
    elif args.command == 'categories':
        print("Available categories:")
        for i, category in enumerate(CATEGORIES, 1):
            print(f"  {i}. {category}")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()