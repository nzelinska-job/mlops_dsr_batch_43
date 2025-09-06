#!/usr/bin/env python3
"""
Test configuration loading and validation.

Run this script to verify your configuration is set up correctly.
"""

import os
import sys
from pathlib import Path

def test_env_file():
    """Test if .env file exists and contains basic variables."""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Create .env file with required variables")
        return False
    
    print("✅ .env file found")
    
    # Read .env content
    with open(env_file, 'r') as f:
        content = f.read()
    
    required_vars = [
        'WANDB_API_KEY',
        'WANDB_ORG', 
        'WANDB_PROJECT',
        'WANDB_MODEL_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in content or f"{var}=your_" in content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing or placeholder values for: {', '.join(missing_vars)}")
        return False
    
    print("✅ Basic environment variables found in .env")
    return True


def test_config_import():
    """Test if config module can be imported."""
    try:
        from app.config import get_settings, get_wandb_config
        print("✅ Config module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        return False


def test_settings_loading():
    """Test settings loading."""
    try:
        from app.config import get_settings
        
        settings = get_settings()
        print("✅ Settings loaded successfully")
        
        print(f"   Environment: {settings.environment}")
        print(f"   Debug: {settings.debug}")
        print(f"   Models dir: {settings.model.models_dir}")
        print(f"   API port: {settings.api.port}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return False


def test_wandb_config():
    """Test W&B specific configuration."""
    try:
        from app.config import get_wandb_config
        
        wandb_config = get_wandb_config()
        
        # Test if we can access config without errors
        org = wandb_config.org
        project = wandb_config.project
        model_name = wandb_config.model_name
        
        print("✅ W&B configuration loaded")
        print(f"   Organization: {org}")
        print(f"   Project: {project}")
        print(f"   Model: {model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ W&B configuration failed: {e}")
        print("   Make sure WANDB_* variables are set in .env")
        return False


def test_model_directories():
    """Test if model directories are created."""
    try:
        from app.config import get_model_config
        
        model_config = get_model_config()
        models_dir = model_config.models_dir
        
        if models_dir.exists():
            print(f"✅ Models directory exists: {models_dir}")
        else:
            print(f"⚠️  Models directory will be created: {models_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model directory test failed: {e}")
        return False


def main():
    """Run all configuration tests."""
    print("🧪 Testing Configuration Setup")
    print("=" * 50)
    
    tests = [
        ("Environment file", test_env_file),
        ("Config import", test_config_import),
        ("Settings loading", test_settings_loading),
        ("W&B configuration", test_wandb_config),
        ("Model directories", test_model_directories)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            # Add helpful hints for common failures
            if test_name == "Environment file":
                print("   💡 Copy the .env example and fill in your values")
            elif test_name == "W&B configuration":
                print("   💡 Get your API key from https://wandb.ai/authorize")
    
    print("\n" + "=" * 50)
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 Configuration is ready!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())