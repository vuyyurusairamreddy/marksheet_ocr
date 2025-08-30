import os
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional
import hashlib
from datetime import datetime
import streamlit as st

from config import Config

logger = logging.getLogger(__name__)

class FileUtils:
    """Utility class for file handling operations"""
    
    def __init__(self):
        self.upload_dir = Path(Config.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        logger.info(f"FileUtils initialized with upload directory: {self.upload_dir}")
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded Streamlit file to temporary location"""
        
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
            
            # Get file extension
            file_extension = self._get_file_extension(uploaded_file.name)
            
            # Create unique filename
            filename = f"{timestamp}_{file_hash}{file_extension}"
            file_path = self.upload_dir / filename
            
            # Validate file size
            file_size = len(uploaded_file.getvalue())
            if file_size > Config.MAX_FILE_SIZE:
                raise ValueError(f"File size ({file_size} bytes) exceeds maximum allowed size ({Config.MAX_FILE_SIZE} bytes)")
            
            # Validate file extension
            if not Config.is_allowed_extension(uploaded_file.name):
                raise ValueError(f"File type not allowed. Supported types: {Config.ALLOWED_EXTENSIONS}")
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            logger.info(f"File saved successfully: {file_path} ({file_size} bytes)")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise
    
    def cleanup_file(self, file_path: str) -> bool:
        """Clean up temporary file"""
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
                return True
            else:
                logger.warning(f"File not found for cleanup: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {e}")
            return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up files older than specified hours"""
        
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            for file_path in self.upload_dir.glob("*"):
                if file_path.is_file():
                    # Get file modification time
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    age_hours = (current_time - file_mtime).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            logger.debug(f"Cleaned up old file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up {file_path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old files")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def validate_file(self, file_path: str) -> dict:
        """Validate file and return information"""
        
        try:
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File not found"}
            
            # Get file stats
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            
            # Check file size
            if file_size > Config.MAX_FILE_SIZE:
                return {
                    "valid": False, 
                    "error": f"File too large ({file_size} bytes > {Config.MAX_FILE_SIZE} bytes)"
                }
            
            # Check file extension
            file_extension = self._get_file_extension(file_path)
            if file_extension not in Config.ALLOWED_EXTENSIONS:
                return {
                    "valid": False,
                    "error": f"Unsupported file type: {file_extension}"
                }
            
            # Get file type info
            file_info = self._get_file_type_info(file_path)
            
            return {
                "valid": True,
                "size": file_size,
                "extension": file_extension,
                "type_info": file_info,
                "path": file_path
            }
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return {"valid": False, "error": str(e)}
    
    def get_file_info(self, file_path: str) -> dict:
        """Get detailed file information"""
        
        try:
            if not os.path.exists(file_path):
                return {"error": "File not found"}
            
            file_stat = os.stat(file_path)
            path_obj = Path(file_path)
            
            return {
                "filename": path_obj.name,
                "size_bytes": file_stat.st_size,
                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                "extension": path_obj.suffix,
                "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "is_image": self._is_image_file(file_path),
                "is_pdf": self._is_pdf_file(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {"error": str(e)}
    
    def create_temp_copy(self, source_path: str) -> str:
        """Create a temporary copy of a file"""
        
        try:
            # Create temporary file with same extension
            source_extension = self._get_file_extension(source_path)
            
            with tempfile.NamedTemporaryFile(suffix=source_extension, delete=False) as temp_file:
                shutil.copy2(source_path, temp_file.name)
                logger.debug(f"Created temporary copy: {temp_file.name}")
                return temp_file.name
                
        except Exception as e:
            logger.error(f"Error creating temporary copy of {source_path}: {e}")
            raise
    
    def ensure_directory_exists(self, directory_path: str) -> bool:
        """Ensure directory exists, create if necessary"""
        
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory_path}: {e}")
            return False
    
    def get_upload_stats(self) -> dict:
        """Get statistics about upload directory"""
        
        try:
            if not self.upload_dir.exists():
                return {"error": "Upload directory not found"}
            
            files = list(self.upload_dir.glob("*"))
            total_files = len([f for f in files if f.is_file()])
            
            if total_files == 0:
                return {
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "total_size_mb": 0,
                    "file_types": {}
                }
            
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            # Count by file type
            file_types = {}
            for file_path in files:
                if file_path.is_file():
                    extension = file_path.suffix.lower()
                    file_types[extension] = file_types.get(extension, 0) + 1
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "directory_path": str(self.upload_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting upload stats: {e}")
            return {"error": str(e)}
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        return Path(filename).suffix.lower()
    
    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is an image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return self._get_file_extension(file_path) in image_extensions
    
    def _is_pdf_file(self, file_path: str) -> bool:
        """Check if file is a PDF"""
        return self._get_file_extension(file_path) == '.pdf'
    
    def _get_file_type_info(self, file_path: str) -> dict:
        """Get detailed file type information"""
        
        extension = self._get_file_extension(file_path)
        
        if extension in {'.jpg', '.jpeg'}:
            return {"type": "image", "format": "JPEG", "description": "JPEG Image"}
        elif extension == '.png':
            return {"type": "image", "format": "PNG", "description": "PNG Image"}
        elif extension == '.pdf':
            return {"type": "document", "format": "PDF", "description": "PDF Document"}
        else:
            return {"type": "unknown", "format": extension.upper(), "description": f"{extension.upper()} File"}
    
    def get_safe_filename(self, filename: str) -> str:
        """Generate a safe filename by removing/replacing problematic characters"""
        
        # Remove path components
        safe_name = os.path.basename(filename)
        
        # Replace problematic characters
        import re
        safe_name = re.sub(r'[^\w\.-]', '_', safe_name)
        
        # Remove multiple underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # Ensure it's not empty
        if not safe_name or safe_name == '_':
            safe_name = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return safe_name
    
    def batch_cleanup(self, file_paths: list) -> dict:
        """Clean up multiple files"""
        
        results = {"success": [], "failed": []}
        
        for file_path in file_paths:
            try:
                if self.cleanup_file(file_path):
                    results["success"].append(file_path)
                else:
                    results["failed"].append(file_path)
            except Exception as e:
                logger.error(f"Error in batch cleanup for {file_path}: {e}")
                results["failed"].append(file_path)
        
        return results