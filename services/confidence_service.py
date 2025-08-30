import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import statistics
from datetime import datetime
import json

from config import Config

logger = logging.getLogger(__name__)

class ConfidenceService:
    """Service for calculating confidence scores for extracted data"""
    
    def __init__(self):
        self.min_threshold = Config.MIN_CONFIDENCE_THRESHOLD
        self.high_threshold = Config.HIGH_CONFIDENCE_THRESHOLD
        
        # Common patterns for validation
        self.patterns = {
            'roll_number': [
                r'^\d{4,12}$',  # Numeric roll numbers
                r'^[A-Z]{1,3}\d{4,10}$',  # Alphanumeric roll numbers
                r'^\d{2}[A-Z]{2,3}\d{4,8}$'  # Mixed format
            ],
            'name': [
                r'^[A-Z][a-z]+ [A-Z][a-z]+',  # First Last
                r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+',  # First Middle Last
                r'^[A-Z\s]{3,50}$'  # All caps names
            ],
            'date': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',  # DD/MM/YYYY or DD-MM-YYYY
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2}',   # DD/MM/YY
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'    # YYYY/MM/DD
            ],
            'marks': [
                r'^\d{1,3}$',  # Simple marks
                r'^\d{1,3}/\d{1,3}$',  # Marks out of total
                r'^\d{1,3}\.\d{1,2}$'  # Decimal marks
            ],
            'grade': [
                r'^[A-F][+\-]?$',  # Letter grades with optional +/-
                r'^(Pass|Fail|PASS|FAIL)$',  # Pass/Fail
                r'^[0-9]\.[0-9]$'  # GPA format
            ],
            'percentage': [
                r'^\d{1,3}\.\d{1,2}%?$',  # Percentage with optional %
                r'^\d{1,3}%$'  # Whole number percentage
            ]
        }
    
    def add_confidence_scores(self, structured_data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        """Add confidence scores to all extracted fields"""
        
        try:
            # Create a copy to avoid modifying original data
            result = {
                "success": True,
                "extracted_data": {},
                "confidence_method": "multi_factor_scoring",
                "processing_time": 0,
                "metadata": {
                    "extraction_timestamp": datetime.now().isoformat(),
                    "confidence_algorithm_version": "1.0",
                    "raw_text_length": len(raw_text)
                }
            }
            
            # Process each section
            for section_name, section_data in structured_data.items():
                if isinstance(section_data, dict):
                    result["extracted_data"][section_name] = self._process_section(
                        section_data, raw_text, section_name
                    )
                elif isinstance(section_data, list):
                    result["extracted_data"][section_name] = self._process_list_section(
                        section_data, raw_text, section_name
                    )
                else:
                    # Handle primitive values
                    result["extracted_data"][section_name] = {
                        "value": section_data,
                        "confidence": self._calculate_basic_confidence(str(section_data), raw_text)
                    }
            
            logger.info("Confidence scores added to all fields")
            return result
            
        except Exception as e:
            logger.error(f"Error adding confidence scores: {e}")
            # Return original data with basic structure
            return {
                "success": False,
                "error": str(e),
                "extracted_data": structured_data,
                "confidence_method": "error_fallback"
            }
    
    def _process_section(self, section_data: Dict[str, Any], raw_text: str, section_name: str) -> Dict[str, Any]:
        """Process a dictionary section and add confidence scores"""
        
        processed_section = {}
        
        for field_name, field_value in section_data.items():
            if field_value is None:
                processed_section[field_name] = {
                    "value": None,
                    "confidence": 0.0
                }
            else:
                confidence = self._calculate_field_confidence(
                    field_name, str(field_value), raw_text, section_name
                )
                
                processed_section[field_name] = {
                    "value": field_value,
                    "confidence": confidence
                }
        
        return processed_section
    
    def _process_list_section(self, section_data: List[Any], raw_text: str, section_name: str) -> List[Dict[str, Any]]:
        """Process a list section (like subjects) and add confidence scores"""
        
        processed_list = []
        
        for item in section_data:
            if isinstance(item, dict):
                processed_item = self._process_section(item, raw_text, section_name)
                processed_list.append(processed_item)
            else:
                # Handle primitive values in list
                confidence = self._calculate_basic_confidence(str(item), raw_text)
                processed_list.append({
                    "value": item,
                    "confidence": confidence
                })
        
        return processed_list
    
    def _calculate_field_confidence(self, field_name: str, field_value: str, raw_text: str, section_name: str) -> float:
        """Calculate confidence score for a specific field"""
        
        if not field_value or field_value.lower() == 'none':
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Pattern matching confidence
        pattern_confidence = self._get_pattern_confidence(field_name, field_value)
        confidence_factors.append(('pattern', pattern_confidence, 0.3))
        
        # Factor 2: Presence in raw text confidence
        text_presence_confidence = self._get_text_presence_confidence(field_value, raw_text)
        confidence_factors.append(('text_presence', text_presence_confidence, 0.25))
        
        # Factor 3: Field-specific validation
        validation_confidence = self._get_validation_confidence(field_name, field_value)
        confidence_factors.append(('validation', validation_confidence, 0.2))
        
        # Factor 4: Context relevance
        context_confidence = self._get_context_confidence(field_name, field_value, section_name)
        confidence_factors.append(('context', context_confidence, 0.15))
        
        # Factor 5: Length and format appropriateness
        format_confidence = self._get_format_confidence(field_name, field_value)
        confidence_factors.append(('format', format_confidence, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in confidence_factors)
        
        # Apply bonuses and penalties
        final_confidence = self._apply_confidence_adjustments(
            total_score, field_name, field_value, raw_text
        )
        
        # Ensure confidence is within valid range
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        logger.debug(f"Confidence for {field_name}='{field_value}': {final_confidence:.3f}")
        
        return round(final_confidence, 3)
    
    def _get_pattern_confidence(self, field_name: str, field_value: str) -> float:
        """Get confidence based on pattern matching"""
        
        # Map field names to pattern categories
        field_pattern_map = {
            'name': 'name',
            'father_name': 'name',
            'mother_name': 'name',
            'father_mother_name': 'name',
            'roll_number': 'roll_number',
            'registration_number': 'roll_number',
            'date_of_birth': 'date',
            'issue_date': 'date',
            'obtained_marks': 'marks',
            'max_marks': 'marks',
            'grade': 'grade',
            'percentage': 'percentage'
        }
        
        pattern_category = field_pattern_map.get(field_name.lower())
        if not pattern_category or pattern_category not in self.patterns:
            return 0.5  # Neutral confidence for unknown patterns
        
        patterns = self.patterns[pattern_category]
        
        # Check if value matches any pattern
        for pattern in patterns:
            if re.match(pattern, field_value, re.IGNORECASE):
                return 0.9  # High confidence for pattern match
        
        return 0.3  # Low confidence for no pattern match
    
    def _get_text_presence_confidence(self, field_value: str, raw_text: str) -> float:
        """Get confidence based on presence in original text"""
        
        if not field_value or len(field_value) < 2:
            return 0.1
        
        # Exact match (case insensitive)
        if field_value.lower() in raw_text.lower():
            return 1.0
        
        # Partial match (word by word)
        words = field_value.split()
        matched_words = sum(1 for word in words if word.lower() in raw_text.lower())
        
        if len(words) > 0:
            partial_match_ratio = matched_words / len(words)
            return 0.3 + (partial_match_ratio * 0.5)  # 0.3 to 0.8 range
        
        return 0.2
    
    def _get_validation_confidence(self, field_name: str, field_value: str) -> float:
        """Get confidence based on field-specific validation"""
        
        field_name_lower = field_name.lower()
        
        # Date validation
        if 'date' in field_name_lower:
            return self._validate_date_format(field_value)
        
        # Name validation
        if 'name' in field_name_lower:
            return self._validate_name_format(field_value)
        
        # Number validation
        if any(keyword in field_name_lower for keyword in ['roll', 'registration', 'number']):
            return self._validate_number_format(field_value)
        
        # Marks validation
        if 'marks' in field_name_lower:
            return self._validate_marks_format(field_value)
        
        # Grade validation
        if 'grade' in field_name_lower:
            return self._validate_grade_format(field_value)
        
        # Year validation
        if 'year' in field_name_lower:
            return self._validate_year_format(field_value)
        
        return 0.5  # Neutral confidence for unknown validation
    
    def _validate_date_format(self, value: str) -> float:
        """Validate date format"""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                try:
                    # Additional validation: try to parse as date
                    parts = re.split('[/-]', value)
                    if len(parts) == 3:
                        day, month, year = map(int, parts)
                        if 1 <= day <= 31 and 1 <= month <= 12:
                            if len(parts[2]) == 2:  # 2-digit year
                                year = 2000 + year if year < 50 else 1900 + year
                            if 1950 <= year <= 2030:
                                return 0.9
                except:
                    pass
                return 0.6  # Pattern matches but validation failed
        
        return 0.1
    
    def _validate_name_format(self, value: str) -> float:
        """Validate name format"""
        if len(value) < 2:
            return 0.1
        
        # Check for reasonable length
        if len(value) > 100:
            return 0.2
        
        # Check for alphabetic characters and spaces
        if re.match(r'^[A-Za-z\s\.]+$', value):
            words = value.split()
            if 1 <= len(words) <= 5:  # Reasonable number of words
                return 0.8
        
        return 0.3
    
    def _validate_number_format(self, value: str) -> float:
        """Validate number format (roll numbers, etc.)"""
        if re.match(r'^[A-Z0-9]{3,15}$', value, re.IGNORECASE):
            return 0.9
        if re.match(r'^\d{4,12}$', value):
            return 0.95
        if len(value) < 3:
            return 0.1
        
        return 0.4
    
    def _validate_marks_format(self, value: str) -> float:
        """Validate marks format"""
        # Numeric marks
        if re.match(r'^\d{1,3}$', value):
            marks = int(value)
            if 0 <= marks <= 100:
                return 0.95
            elif marks <= 200:
                return 0.8
        
        # Marks with total (e.g., "85/100")
        if re.match(r'^\d{1,3}/\d{1,3}$', value):
            try:
                obtained, total = map(int, value.split('/'))
                if 0 <= obtained <= total <= 200:
                    return 0.9
            except:
                pass
        
        return 0.3
    
    def _validate_grade_format(self, value: str) -> float:
        """Validate grade format"""
        # Letter grades
        if re.match(r'^[A-F][+\-]?$', value, re.IGNORECASE):
            return 0.9
        
        # Pass/Fail
        if value.upper() in ['PASS', 'FAIL', 'P', 'F']:
            return 0.85
        
        # GPA format
        if re.match(r'^\d\.\d{1,2}$', value):
            try:
                gpa = float(value)
                if 0.0 <= gpa <= 10.0:
                    return 0.9
            except:
                pass
        
        return 0.3
    
    def _validate_year_format(self, value: str) -> float:
        """Validate year format"""
        if re.match(r'^\d{4}$', value):
            year = int(value)
            if 1950 <= year <= 2030:
                return 0.95
        
        return 0.2
    
    def _get_context_confidence(self, field_name: str, field_value: str, section_name: str) -> float:
        """Get confidence based on contextual appropriateness"""
        
        # Higher confidence for appropriate section-field combinations
        appropriate_combinations = {
            'candidate_details': ['name', 'roll', 'date', 'year', 'board', 'institution'],
            'subjects': ['subject', 'marks', 'grade', 'credits'],
            'result_info': ['grade', 'marks', 'percentage', 'cgpa', 'division'],
            'document_info': ['date', 'place', 'number', 'authority']
        }
        
        if section_name in appropriate_combinations:
            for keyword in appropriate_combinations[section_name]:
                if keyword in field_name.lower():
                    return 0.8
        
        return 0.5
    
    def _get_format_confidence(self, field_name: str, field_value: str) -> float:
        """Get confidence based on format appropriateness"""
        
        # Length-based confidence
        length = len(field_value)
        
        # Very short values are suspicious
        if length < 2:
            return 0.1
        
        # Very long values might be extraction errors
        if length > 100:
            return 0.2
        
        # Reasonable length gets good confidence
        if 2 <= length <= 50:
            return 0.8
        
        return 0.5
    
    def _apply_confidence_adjustments(self, base_confidence: float, field_name: str, field_value: str, raw_text: str) -> float:
        """Apply final adjustments to confidence score"""
        
        adjusted = base_confidence
        
        # Bonus for commonly expected fields
        important_fields = ['name', 'roll_number', 'marks', 'grade']
        if any(keyword in field_name.lower() for keyword in important_fields):
            adjusted += 0.05
        
        # Penalty for suspicious patterns
        if re.search(r'[^\w\s\.\-/]', field_value):  # Special characters
            adjusted -= 0.1
        
        # Penalty for repeated characters (OCR errors)
        if re.search(r'(.)\1{3,}', field_value):  # 4+ repeated chars
            adjusted -= 0.15
        
        # Bonus for numeric consistency in marks
        if 'marks' in field_name.lower() and field_value.isdigit():
            adjusted += 0.05
        
        return adjusted
    
    def _calculate_basic_confidence(self, field_value: str, raw_text: str) -> float:
        """Calculate basic confidence for unknown field types"""
        
        if not field_value:
            return 0.0
        
        # Simple presence-based confidence
        if field_value.lower() in raw_text.lower():
            return 0.7
        
        # Partial match
        words = field_value.split()
        matched_words = sum(1 for word in words if word.lower() in raw_text.lower())
        
        if len(words) > 0:
            return 0.3 + (matched_words / len(words)) * 0.4
        
        return 0.3
    
    def get_confidence_summary(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate confidence summary statistics"""
        
        confidences = []
        
        def extract_confidences(data):
            if isinstance(data, dict):
                if 'confidence' in data:
                    confidences.append(data['confidence'])
                else:
                    for value in data.values():
                        extract_confidences(value)
            elif isinstance(data, list):
                for item in data:
                    extract_confidences(item)
        
        extract_confidences(processed_data.get('extracted_data', {}))
        
        if confidences:
            return {
                'total_fields': len(confidences),
                'average_confidence': statistics.mean(confidences),
                'median_confidence': statistics.median(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences),
                'high_confidence_count': sum(1 for c in confidences if c >= self.high_threshold),
                'low_confidence_count': sum(1 for c in confidences if c < self.min_threshold),
                'confidence_distribution': {
                    'high': sum(1 for c in confidences if c >= self.high_threshold),
                    'medium': sum(1 for c in confidences if self.min_threshold <= c < self.high_threshold),
                    'low': sum(1 for c in confidences if c < self.min_threshold)
                }
            }
        
        return {'total_fields': 0, 'average_confidence': 0.0}