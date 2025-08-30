import os
from dotenv import load_dotenv
from typing import Set

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Marksheet Extraction API"""
    
    # Application Information
    APP_NAME = "Marksheet Extraction API"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "AI-powered marksheet data extraction with confidence scoring"
    
    # File Handling Configuration
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024  # Convert MB to bytes
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".pdf"}
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp_uploads")
    
    # API Configuration
    API_KEY = os.getenv("API_KEY", "default-api-key-change-in-production")
    
    # Perplexity API Configuration
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    PERPLEXITY_MODEL = "sonar-pro"
    PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
    
    # OCR Configuration
    TESSERACT_CMD = os.getenv("TESSERACT_CMD")  # Optional: set if tesseract not in PATH
    # OCR Configuration
    #USE_GPU_FOR_OCR = bool(os.getenv("USE_GPU_FOR_OCR", "False").lower() == "true")
    
    # Confidence Scoring Configuration
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.3"))
    HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.8"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Processing Configuration
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes
    
    # LLM Configuration
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_DELAY = int(os.getenv("LLM_RETRY_DELAY", "2"))  # seconds
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate required configuration parameters"""
        errors = []
        
        if not cls.PERPLEXITY_API_KEY:
            errors.append("PERPLEXITY_API_KEY is required")
        
        if cls.MAX_FILE_SIZE <= 0:
            errors.append("MAX_FILE_SIZE must be positive")
        
        if not (0 <= cls.MIN_CONFIDENCE_THRESHOLD <= 1):
            errors.append("MIN_CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if not (0 <= cls.HIGH_CONFIDENCE_THRESHOLD <= 1):
            errors.append("HIGH_CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if cls.MIN_CONFIDENCE_THRESHOLD >= cls.HIGH_CONFIDENCE_THRESHOLD:
            errors.append("MIN_CONFIDENCE_THRESHOLD must be less than HIGH_CONFIDENCE_THRESHOLD")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        
        return True
    
    @classmethod
    def get_file_size_mb(cls) -> float:
        """Get max file size in MB"""
        return cls.MAX_FILE_SIZE / (1024 * 1024)
    
    @classmethod
    def is_allowed_extension(cls, filename: str) -> bool:
        """Check if file extension is allowed"""
        return any(filename.lower().endswith(ext) for ext in cls.ALLOWED_EXTENSIONS)
    
    @classmethod
    def get_upload_path(cls, filename: str) -> str:
        """Get full upload path for a file"""
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        return os.path.join(cls.UPLOAD_DIR, filename)

# Validate configuration on import
try:
    Config.validate_config()
except ValueError as e:
    print(f"Warning: {e}")

# Export commonly used values
MAX_FILE_SIZE = Config.MAX_FILE_SIZE
ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS
UPLOAD_DIR = Config.UPLOAD_DIR