import httpx
import json
import logging
import asyncio
import re
from typing import Dict, Any, Optional
import time
from datetime import datetime

from config import Config

logger = logging.getLogger(__name__)

class LLMService:
    """Service for processing extracted text using Perplexity's LLM"""
    
    def __init__(self):
        self.api_key = Config.PERPLEXITY_API_KEY
        self.model = Config.PERPLEXITY_MODEL
        self.base_url = Config.PERPLEXITY_BASE_URL
        self.max_retries = Config.LLM_MAX_RETRIES
        self.retry_delay = Config.LLM_RETRY_DELAY
        
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY is required but not set")
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=Config.REQUEST_TIMEOUT
        )
    
    def __del__(self):
        """Cleanup HTTP client"""
        if hasattr(self, 'client'):
            asyncio.create_task(self.client.aclose())
    
    def extract_structured_data(self, raw_text: str) -> Dict[str, Any]:
        """Extract structured data from raw OCR text using LLM"""
        try:
            # Preprocess the text first
            processed_text = self._preprocess_ocr_text(raw_text)
            
            # Run async function in sync context
            return asyncio.run(self._extract_structured_data_async(processed_text))
        except Exception as e:
            logger.error(f"Error in structured data extraction: {e}")
            raise
    
    def _preprocess_ocr_text(self, raw_text: str) -> str:
        """Preprocess OCR text to improve extraction accuracy"""
        
        # Dictionary for converting written numbers to digits
        number_words = {
            'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
            'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
            'TEN': '10', 'ELEVEN': '11', 'TWELVE': '12', 'THIRTEEN': '13',
            'FOURTEEN': '14', 'FIFTEEN': '15', 'SIXTEEN': '16', 'SEVENTEEN': '17',
            'EIGHTEEN': '18', 'NINETEEN': '19', 'TWENTY': '20', 'THIRTY': '30'
        }
        
        # Month names to numbers
        months = {
            'JANUARY': '01', 'FEBRUARY': '02', 'MARCH': '03', 'APRIL': '04',
            'MAY': '05', 'JUNE': '06', 'JULY': '07', 'AUGUST': '08',
            'SEPTEMBER': '09', 'OCTOBER': '10', 'NOVEMBER': '11', 'DECEMBER': '12'
        }
        
        processed_text = raw_text.upper()
        
        # Add hints for better extraction
        hints = []
        
        # Look for roll number patterns
        roll_pattern = r'bearing Roll No\.\s*([A-Z\s]*(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|ZERO|\d)[A-Z\s\d]*)'
        roll_match = re.search(roll_pattern, processed_text, re.IGNORECASE)
        if roll_match:
            written_number = roll_match.group(1).strip()
            # Convert written numbers to digits
            digits = []
            words = written_number.split()
            for word in words:
                word = word.strip().upper()
                if word in number_words:
                    digits.append(number_words[word])
                elif word.isdigit():
                    digits.append(word)
            if digits:
                hint_roll = ''.join(digits)
                hints.append(f"EXTRACTION_HINT: Roll number detected as: {hint_roll}")
        
        # Look for numeric roll numbers directly
        numeric_roll_pattern = r'bearing Roll No\.\s*(\d+)'
        numeric_roll_match = re.search(numeric_roll_pattern, processed_text, re.IGNORECASE)
        if numeric_roll_match:
            hints.append(f"EXTRACTION_HINT: Roll number detected as: {numeric_roll_match.group(1)}")
        
        # Pattern for date of birth
        date_pattern = r'DATE OF BIRTH\s+([A-Z\s\d]*(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)[A-Z\s\d]*)'
        date_match = re.search(date_pattern, processed_text, re.IGNORECASE)
        if date_match:
            written_date = date_match.group(1).strip()
            # Try to extract day, month, year
            words = written_date.split()
            day = None
            month = None
            year_parts = []
            
            for i, word in enumerate(words):
                word = word.upper()
                # Check for day
                if word in number_words and not day:
                    day = number_words[word].zfill(2)
                elif word.isdigit() and len(word) <= 2 and not day:
                    day = word.zfill(2)
                # Check for month
                elif word in months:
                    month = months[word]
                    # Look for year after month
                    remaining_words = words[i+1:]
                    for remaining_word in remaining_words:
                        remaining_word = remaining_word.upper()
                        if remaining_word in number_words:
                            year_parts.append(number_words[remaining_word])
                        elif remaining_word.isdigit():
                            year_parts.append(remaining_word)
                    break
            
            if day and month and len(year_parts) >= 4:
                year = ''.join(year_parts[:4])
                hints.append(f"EXTRACTION_HINT: Date of birth detected as: {day}/{month}/{year}")
        
        # Look for numeric dates directly (DD/MM/YYYY or DD-MM-YYYY)
        numeric_date_pattern = r'DATE OF BIRTH[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})'
        numeric_date_match = re.search(numeric_date_pattern, processed_text, re.IGNORECASE)
        if numeric_date_match:
            hints.append(f"EXTRACTION_HINT: Date of birth detected as: {numeric_date_match.group(1)}")
        
        # Look for GPA/CGPA
        gpa_pattern = r'(?:GPA|CGPA)\s*[:\-]?\s*(\d+\.?\d*)'
        gpa_match = re.search(gpa_pattern, processed_text, re.IGNORECASE)
        if gpa_match:
            hints.append(f"EXTRACTION_HINT: GPA/CGPA detected as: {gpa_match.group(1)}")
        
        # Look for examination year
        exam_year_pattern = r'(?:MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{4})|EXAMINATION.*?(\d{4})|held in.*?(\d{4})'
        exam_year_match = re.search(exam_year_pattern, processed_text, re.IGNORECASE)
        if exam_year_match:
            year = exam_year_match.group(1) or exam_year_match.group(2) or exam_year_match.group(3)
            hints.append(f"EXTRACTION_HINT: Examination year detected as: {year}")
        
        # Add hints to the text
        if hints:
            processed_text = processed_text + "\n\nEXTRACTION_HINTS:\n" + "\n".join(hints)
        
        return processed_text
    
    async def _extract_structured_data_async(self, raw_text: str) -> Dict[str, Any]:
        """Async implementation of structured data extraction"""
        
        prompt = self._create_extraction_prompt(raw_text)
        
        for attempt in range(self.max_retries):
            try:
                response = await self._call_llm(prompt)
                structured_data = self._parse_llm_response(response)
                return structured_data
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
    
    def _create_extraction_prompt(self, raw_text: str) -> str:
        """Create a comprehensive prompt for data extraction"""
        
        prompt = f"""
You are an AI expert in extracting structured data from educational marksheets/transcripts. 
Analyze the following OCR-extracted text from a marksheet and extract information in the exact JSON format specified below.

OCR TEXT TO ANALYZE:
{raw_text}

CRITICAL INSTRUCTIONS FOR COMMON OCR ISSUES:

1. **DATE CONVERSION**: Convert written dates to DD/MM/YYYY format:
   - "ONE FIVE NOVEMBER TWO ZERO ZERO TWO" → "15/11/2002"
   - "FIFTEEN NOVEMBER TWO THOUSAND TWO" → "15/11/2002"  
   - "1/11/2002" → "01/11/2002"
   - Look for patterns like "DATE OF BIRTH" followed by written numbers/months

2. **NUMBER EXTRACTION**: Convert written numbers to digits:
   - "ONE NINE ONE THREE ONE THREE THREE EIGHT ONE SIX" → "1913133816"
   - Look for long sequences of written numbers, especially near "Roll No" or "bearing Roll No"
   - Common patterns: "bearing Roll No. [NUMBER]" or "Roll Number [NUMBER]"

3. **ROLL/REGISTRATION NUMBERS**: Look carefully for:
   - "bearing Roll No." followed by numbers (often 10+ digits)
   - "Registration Number" or "Reg No"
   - Numbers that appear after student identification sections
   - May appear as both written words and digits

4. **NAME EXTRACTION**: 
   - Student name often appears after "CERTIFIED THAT" or similar phrases
   - Father/Mother names appear after "FATHER NAME" or "MOTHER NAME"
   - Handle names in ALL CAPS format

5. **GRADE/MARKS PATTERNS**:
   - Look for subject tables with columns like "GRADE", "Internal", "External"
   - Common grades: A1, A2, B1, B2, C1, C2, etc.
   - Grade points: Usually single/double digits (09, 10, etc.)

6. **GPA/CGPA DETECTION**:
   - Look for "GPA", "CGPA" followed by decimal numbers
   - Often appears in result summary sections

7. **EXAMINATION YEAR**:
   - Look for month-year combinations like "MARCH 2019"
   - Check phrases like "examination held in", "appeared and PASSED"

8. **PAY ATTENTION TO EXTRACTION HINTS**: 
   - If you see "EXTRACTION_HINT" in the text, use that information to guide your extraction

REQUIRED JSON OUTPUT FORMAT:
{{
  "candidate_details": {{
    "name": "Full student name (e.g., VUYYURU SAIRAMREDDY)",
    "father_mother_name": "Parent name (e.g., SUDHAKARREDDY, HEMALATHA)",
    "roll_number": "Roll/Enrollment number (convert written numbers to digits)",
    "registration_number": "Registration number if different from roll",
    "date_of_birth": "DOB in DD/MM/YYYY format (convert written dates)",
    "exam_year": "Examination year (e.g., 2019)",
    "board_university": "Board/University name (e.g., Board of Secondary Education, Andhra Pradesh)",
    "institution": "School/College name"
  }},
  "subjects": [
    {{
      "subject_name": "Subject name (e.g., FIRST LANGUAGE, MATHEMATICS)",
      "max_marks": "Maximum marks or credits",
      "obtained_marks": "Obtained marks or credits", 
      "grade": "Grade (e.g., A2, A1)",
      "credits": "Grade points (e.g., 09, 10)"
    }}
  ],
  "result_info": {{
    "overall_grade": "Overall grade/division/class",
    "total_marks": "Total marks in format 'obtained/maximum'",
    "percentage": "Percentage if mentioned",
    "cgpa": "CGPA/GPA if mentioned (e.g., 9.5)",
    "division": "Division (First/Second/Third) if mentioned",
    "pass_status": "PASSED/FAILED status"
  }},
  "document_info": {{
    "issue_date": "Date of issue",
    "issue_place": "Place of issue",
    "document_number": "Certificate/Document number (e.g., SS 228786)",
    "signature_authority": "Signing authority name"
  }}
}}

EXAMPLE TRANSFORMATIONS FOR YOUR REFERENCE:
- "bearing Roll No. 1913133816" → "roll_number": "1913133816"
- "DATE OF BIRTH ONE FIVE NOVEMBER TWO ZERO ZERO TWO" → "date_of_birth": "15/11/2002"
- "MARCH 2019" → "exam_year": "2019"
- "GPA 9.5" → "cgpa": "9.5"
- "CERTIFIED THAT VUYYURU SAIRAMREDDY" → "name": "VUYYURU SAIRAMREDDY"
- "FATHER NAME : SUDHAKARREDDY" → "father_mother_name": "SUDHAKARREDDY"

IMPORTANT NOTES:
- Use null for truly missing information, don't make up data
- For unclear text, provide your best interpretation based on context
- Extract all subjects found in the table format
- Pay special attention to number sequences and date patterns
- Look for the document header for board/institution information
- Use the EXTRACTION_HINTS if provided in the text

Respond with ONLY the JSON object, no additional text or explanations.
"""
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """Make API call to Perplexity LLM"""
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent extraction
            "max_tokens": 4000,
            "top_p": 0.9
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' not in result or len(result['choices']) == 0:
                raise ValueError("Invalid LLM response format")
            
            content = result['choices'][0]['message']['content'].strip()
            logger.info(f"LLM response received: {len(content)} characters")
            
            return content
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling LLM: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON"""
        
        try:
            # Try to find JSON in the response
            response = response.strip()
            
            # Remove any markdown code blocks
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Find JSON object boundaries
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx]
            
            # Parse JSON
            parsed_data = json.loads(json_str)
            
            # Validate structure
            self._validate_parsed_data(parsed_data)
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Return fallback structure
            return self._create_fallback_structure()
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_structure()
    
    def _validate_parsed_data(self, data: Dict[str, Any]) -> None:
        """Validate the structure of parsed data"""
        
        required_sections = ['candidate_details', 'subjects', 'result_info', 'document_info']
        
        for section in required_sections:
            if section not in data:
                logger.warning(f"Missing section in parsed data: {section}")
                data[section] = {}
        
        # Ensure subjects is a list
        if not isinstance(data.get('subjects'), list):
            data['subjects'] = []
        
        logger.info("Data structure validation passed")
    
    def _create_fallback_structure(self) -> Dict[str, Any]:
        """Create fallback structure when LLM parsing fails"""
        
        return {
            "candidate_details": {
                "name": None,
                "father_mother_name": None,
                "roll_number": None,
                "registration_number": None,
                "date_of_birth": None,
                "exam_year": None,
                "board_university": None,
                "institution": None
            },
            "subjects": [],
            "result_info": {
                "overall_grade": None,
                "total_marks": None,
                "percentage": None,
                "cgpa": None,
                "division": None,
                "pass_status": None
            },
            "document_info": {
                "issue_date": None,
                "issue_place": None,
                "document_number": None,
                "signature_authority": None
            }
        }
    
    def enhance_extraction_with_context(self, raw_text: str, ocr_confidence_data: list) -> Dict[str, Any]:
        """Enhanced extraction using OCR confidence data"""
        
        try:
            # Create enhanced prompt with confidence information
            high_confidence_text = [item[0] for item in ocr_confidence_data if item[1] > 0.8]
            medium_confidence_text = [item[0] for item in ocr_confidence_data if 0.5 <= item[1] <= 0.8]
            
            enhanced_prompt = f"""
ORIGINAL TEXT: {raw_text}

HIGH CONFIDENCE WORDS (>80%): {' '.join(high_confidence_text)}
MEDIUM CONFIDENCE WORDS (50-80%): {' '.join(medium_confidence_text)}

Use this confidence information to prioritize more reliable text when extracting data.
Focus on high-confidence words for critical information like names, numbers, and grades.

{self._create_extraction_prompt(raw_text)}
"""
            
            return asyncio.run(self._extract_structured_data_async_enhanced(enhanced_prompt))
            
        except Exception as e:
            logger.error(f"Enhanced extraction failed: {e}")
            # Fallback to regular extraction
            return self.extract_structured_data(raw_text)
    
    async def _extract_structured_data_async_enhanced(self, prompt: str) -> Dict[str, Any]:
        """Enhanced async extraction with confidence data"""
        
        for attempt in range(self.max_retries):
            try:
                response = await self._call_llm(prompt)
                structured_data = self._parse_llm_response(response)
                return structured_data
                
            except Exception as e:
                logger.warning(f"Enhanced LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise