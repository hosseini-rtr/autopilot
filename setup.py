#!/usr/bin/env python
"""
Setup script for the Autopilot project.
Provides utilities for environment setup and initialization.

Supports both UV (recommended) and pip for dependency management.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def detect_package_manager():
    """Detect which package manager is being used."""
    # Check if UV is available
    if shutil.which("uv"):
        return "uv"
    elif shutil.which("pip"):
        return "pip"
    else:
        return None


def create_directories():
    """Create necessary project directories."""
    directories = [
        PROJECT_ROOT / "data" / "driving_log",
        PROJECT_ROOT / "data" / "IMG",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "models",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created/verified directory: {directory}")


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "numpy",
        "pandas",
        "cv2",
        "tensorflow",
        "keras",
        "flask",
        "socketio",
        "eventlet",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.warning(f"✗ {package}")
            missing.append(package)

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        pm = detect_package_manager()
        if pm == "uv":
            logger.info("Install with: uv sync")
        else:
            logger.info("Install with: pip install -r requirements.txt")
        return False

    logger.info("✅ All dependencies installed!")
    return True


def setup_environment():
    """Initialize project environment."""
    logger.info("🚀 Initializing Autopilot Project Environment")

    # Detect package manager
    pm = detect_package_manager()
    if pm:
        logger.info(f"📦 Package manager detected: {pm.upper()}")
    else:
        logger.warning("⚠️  No package manager found. Install UV or pip.")

    logger.info("📁 Setting up directories...")
    create_directories()

    logger.info("📦 Checking dependencies...")
    if not check_dependencies():
        return False

    logger.info("✨ Setup complete! You're ready to go.")
    logger.info("Next steps:")

    pm = detect_package_manager()
    if pm == "uv":
        logger.info("1. UV detected! Dependencies are managed in pyproject.toml")
        logger.info("2. Run UV commands:")
        logger.info("   uv sync              # Install/update dependencies")
        logger.info("   uv run python src/drive.py  # Run application")
        logger.info("3. Or use uv pip for pip-compatible commands:")
        logger.info("   uv pip install package_name")
    else:
        logger.info("1. Place your training data in data/driving_log/")
        logger.info("2. Train a model using notebooks/train_model.ipynb")
        logger.info("3. Run the autonomous driving: python src/drive.py")
        logger.info("\nFor faster dependency management, install UV:")
        logger.info("   curl -LsSf https://astral.sh/uv/install.sh | sh")

    return True


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
