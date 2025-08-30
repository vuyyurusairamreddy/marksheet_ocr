import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Optional, List, Tuple
import os
import io
import re

from config import Config

logger = logging.getLogger(__name__)

class OCRService:
    """Service for extracting text from images and PDFs using OCR"""
    
    def __init__(self):
        # Set Tesseract path if specified in config
        if Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
        
        # Test Tesseract availability
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.error(f"Tesseract initialization failed: {e}")
            raise
    
    def extract_from_image(self, image_path: str) -> str:
        """Extract text from an image file with enhanced processing"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Try multiple preprocessing approaches for better accuracy
            text_results = []
            
            # Method 1: Standard preprocessing
            processed_image1 = self._preprocess_image(image)
            custom_config1 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()/-: '
            text1 = pytesseract.image_to_string(processed_image1, config=custom_config1)
            text_results.append(text1)
            
            # Method 2: Different PSM mode for dense text
            custom_config2 = r'--oem 3 --psm 4'
            text2 = pytesseract.image_to_string(processed_image1, config=custom_config2)
            text_results.append(text2)
            
            # Method 3: Enhanced preprocessing for difficult text
            processed_image2 = self._preprocess_image_enhanced(image)
            custom_config3 = r'--oem 3 --psm 6'
            text3 = pytesseract.image_to_string(processed_image2, config=custom_config3)
            text_results.append(text3)
            
            # Combine results - choose the longest/most comprehensive result
            best_text = max(text_results, key=len)
            
            # Clean and return text
            cleaned_text = self._clean_text(best_text)
            
            # Post-process for common OCR errors in marksheets
            cleaned_text = self._fix_common_ocr_errors(cleaned_text)
            
            logger.info(f"Successfully extracted {len(cleaned_text)} characters from image")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            raise
    
    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            text_content = []
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Try text extraction first (for text-based PDFs)
                text = page.get_text()
                
                if text.strip():
                    text_content.append(text)
                else:
                    # If no text found, use OCR on page image
                    logger.info(f"No text found on page {page_num + 1}, using OCR")
                    
                    # Convert page to image
                    mat = fitz.Matrix(2.0, 2.0)  # Increase resolution
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Convert to OpenCV format
                    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Preprocess and OCR
                    processed_image = self._preprocess_image(cv_image)
                    custom_config = r'--oem 3 --psm 6'
                    ocr_text = pytesseract.image_to_string(processed_image, config=custom_config)
                    text_content.append(ocr_text)
            
            doc.close()
            
            # Combine all pages
            full_text = '\n\n'.join(text_content)
            cleaned_text = self._clean_text(full_text)
            
            # Post-process for common OCR errors
            cleaned_text = self._fix_common_ocr_errors(cleaned_text)
            
            logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF ({len(doc)} pages)")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply different preprocessing techniques
            
            # 1. Noise removal
            denoised = cv2.medianBlur(gray, 3)
            
            # 2. Contrast enhancement using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 3. Threshold the image
            # Use Otsu's thresholding
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 4. Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # 5. Resize image if it's too small (OCR works better with larger images)
            height, width = cleaned.shape
            if height < 500 or width < 500:
                scale_factor = max(500/height, 500/width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return original grayscale as fallback
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    def _preprocess_image_enhanced(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for difficult text recognition"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 1. Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 2. Adaptive threshold for varying lighting
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 3. Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            
            # 4. Dilation to make text thicker and more readable
            dilated = cv2.dilate(morph, kernel, iterations=1)
            
            # 5. Scale up for better OCR
            height, width = dilated.shape
            scale_factor = max(2.0, 800/max(height, width))
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            scaled = cv2.resize(dilated, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return scaled
            
        except Exception as e:
            logger.error(f"Error in enhanced preprocessing: {e}")
            return self._preprocess_image(image)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        # Join lines with single newline
        cleaned = '\n'.join(lines)
        
        # Remove multiple spaces
        cleaned = re.sub(r' +', ' ', cleaned)
        
        return cleaned
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors specific to marksheets"""
        if not text:
            return text
        
        # Common OCR error corrections for marksheets
        corrections = {
            # Common character misreads
            '0': ['O', 'o'],  # Zero vs O
            '1': ['l', 'I', '|'],  # One vs l, I, |
            '5': ['S', 's'],  # Five vs S
            '6': ['G', 'b'],  # Six vs G, b
            '8': ['B'],  # Eight vs B
            '9': ['g', 'q'],  # Nine vs g, q
            
            # Common word corrections for marksheets
            'bearing': ['bearmg', 'bearmg', 'beang'],
            'Roll': ['Rol', 'RolI', 'Roil'],
            'CERTIFICATE': ['CERTIEICATE', 'CERTFICATE'],
            'SECONDARY': ['SECONDAFY', 'SECNDARY'],
            'EDUCATION': ['EDUCATON', 'EDUCATBN'],
            'EXAMINATION': ['EXAMWATION', 'EXAMNATION'],
            'MATHEMATICS': ['MATHERNATICS', 'MATHEMATCS'],
            'ENGLISH': ['ENGLSH', 'ENGLRSH'],
            'SCIENCE': ['SCENCE', 'SCLENCE'],
            'GRADE': ['GFADE', 'GRABE'],
            'MARKS': ['MARRS', 'MAFKS'],
            'FATHER': ['FATHEF', 'FATER'],
            'MOTHER': ['MOTHEF', 'MOTER'],
            'NOVEMBER': ['NOVEMEER', 'NOVEMBEF'],
            'MARCH': ['MAFCH', 'MARCF'],
            'PASSED': ['PASSEB', 'PASSEF'],
            'BOARD': ['BOAFD', 'BOARB']
        }
        
        corrected_text = text
        
        # Apply corrections
        for correct_word, error_variants in corrections.items():
            for error_variant in error_variants:
                corrected_text = re.sub(
                    r'\b' + re.escape(error_variant) + r'\b',
                    correct_word,
                    corrected_text,
                    flags=re.IGNORECASE
                )
        
        # Fix spacing issues around numbers
        corrected_text = re.sub(r'(\d)\s+(\d)', r'\1\2', corrected_text)  # Remove spaces within numbers
        
        # Fix common date format issues
        corrected_text = re.sub(r'(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})', r'\1/\2/\3', corrected_text)
        
        # Fix roll number patterns
        corrected_text = re.sub(r'bearing\s+Roll\s+No\s*\.?\s*(\d+)', r'bearing Roll No. \1', corrected_text, flags=re.IGNORECASE)
        
        return corrected_text
    
    def get_text_with_confidence(self, image_path: str) -> List[Tuple[str, float]]:
        """Extract text with confidence scores for each word"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Try both preprocessing methods and combine results
            processed_image1 = self._preprocess_image(image)
            processed_image2 = self._preprocess_image_enhanced(image)
            
            confidence_data = []
            
            # Get confidence data from both methods
            for processed_image in [processed_image1, processed_image2]:
                data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
                
                n_boxes = len(data['level'])
                for i in range(n_boxes):
                    if int(data['conf'][i]) > 0:  # Only include confident detections
                        text = data['text'][i].strip()
                        if text:  # Only include non-empty text
                            confidence = int(data['conf'][i]) / 100.0  # Convert to 0-1 scale
                            confidence_data.append((text, confidence))
            
            # Remove duplicates and keep highest confidence for each word
            word_confidence_map = {}
            for text, confidence in confidence_data:
                if text not in word_confidence_map or confidence > word_confidence_map[text]:
                    word_confidence_map[text] = confidence
            
            # Convert back to list of tuples
            final_confidence_data = [(text, conf) for text, conf in word_confidence_map.items()]
            
            # Apply OCR error corrections to high-confidence words
            corrected_confidence_data = []
            for text, confidence in final_confidence_data:
                corrected_text = self._fix_common_ocr_errors(text)
                corrected_confidence_data.append((corrected_text, confidence))
            
            return corrected_confidence_data
            
        except Exception as e:
            logger.error(f"Error getting text with confidence: {e}")
            return []
    
    def extract_text_regions(self, image_path: str) -> List[dict]:
        """Extract text with bounding box information"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            processed_image = self._preprocess_image(image)
            
            # Get bounding boxes
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            regions = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Minimum confidence threshold
                    text = data['text'][i].strip()
                    if text:
                        # Apply OCR error corrections
                        corrected_text = self._fix_common_ocr_errors(text)
                        
                        region = {
                            'text': corrected_text,
                            'original_text': text,  # Keep original for reference
                            'confidence': int(data['conf'][i]) / 100.0,
                            'bbox': {
                                'x': data['left'][i],
                                'y': data['top'][i],
                                'width': data['width'][i],
                                'height': data['height'][i]
                            },
                            'level': data['level'][i]
                        }
                        regions.append(region)
            
            return regions
            
        except Exception as e:
            logger.error(f"Error extracting text regions: {e}")
            return []
    
    def extract_structured_regions(self, image_path: str) -> dict:
        """Extract text regions organized by likely content type"""
        try:
            regions = self.extract_text_regions(image_path)
            
            structured_regions = {
                'header': [],
                'candidate_info': [],
                'subjects_table': [],
                'grades': [],
                'footer': [],
                'numbers': []
            }
            
            for region in regions:
                text = region['text'].upper()
                y_position = region['bbox']['y']
                
                # Classify regions based on content and position
                if any(keyword in text for keyword in ['BOARD', 'SECONDARY', 'EDUCATION', 'CERTIFICATE']):
                    structured_regions['header'].append(region)
                elif any(keyword in text for keyword in ['CERTIFIED', 'NAME', 'FATHER', 'MOTHER', 'ROLL', 'BIRTH']):
                    structured_regions['candidate_info'].append(region)
                elif any(keyword in text for keyword in ['SUBJECT', 'GRADE', 'MARKS', 'INTERNAL', 'EXTERNAL']):
                    structured_regions['subjects_table'].append(region)
                elif re.match(r'^[A-C][1-2]$', text.strip()) or text.strip() in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
                    structured_regions['grades'].append(region)
                elif text.isdigit() and len(text) >= 4:
                    structured_regions['numbers'].append(region)
                elif y_position > 0.8:  # Bottom 20% of image
                    structured_regions['footer'].append(region)
            
            return structured_regions
        except Exception as e:
            logger.error(f"Error extracting structured regions: {e}")
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting structured regions: {e}")
            return {}