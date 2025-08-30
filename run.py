#!/usr/bin/env python3
"""
Application runner for Marksheet Extraction API
Provides multiple ways to run the application with proper error handling
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import streamlit
        import PIL
        import cv2
        import pytesseract
        import fitz  # PyMuPDF
        import numpy
        import pandas
        import dotenv
        import httpx
        logger.info("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_tesseract():
    """Check if Tesseract OCR is available"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        logger.info("✅ Tesseract OCR is available")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Tesseract OCR check failed: {e}")
        logger.info("Please install Tesseract OCR:")
        logger.info("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        logger.info("  macOS: brew install tesseract")
        logger.info("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def check_environment():
    """Check environment variables"""
    env_file = Path('.env')
    if not env_file.exists():
        logger.warning("⚠️ .env file not found, using environment variables")
        logger.info("Copy .env.example to .env and configure your API keys")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    perplexity_key = os.getenv('PERPLEXITY_API_KEY')
    if not perplexity_key:
        logger.error("❌ PERPLEXITY_API_KEY not set")
        logger.info("Please set your Perplexity API key in .env file")
        return False
    
    logger.info("✅ Environment configuration looks good")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['temp_uploads', 'tests/sample_marksheets']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("✅ Directories created")

def run_streamlit():
    """Run the Streamlit application"""
    try:
        logger.info("🚀 Starting Marksheet Extraction API...")
        logger.info("📱 Opening web interface at http://localhost:8501")
        
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true"]
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)

def main():
    """Main function to run the application"""
    logger.info("🎓 Marksheet Extraction API - Starting Up...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check Tesseract (warning only, not blocking)
    check_tesseract()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()