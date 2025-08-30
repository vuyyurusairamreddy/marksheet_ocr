# 🎓 Marksheet Data Extractor

A simple Streamlit application that extracts structured data from marksheet images and PDFs using OCR and AI.

## Features

- ✅ Upload marksheet images (JPG, PNG) or PDFs
- 🔍 OCR text extraction with preprocessing
- 🧠 AI-powered structured data extraction using OpenAI GPT
- 📊 Confidence scoring for extracted fields
- 📋 User-friendly display with formatted views
- 💾 JSON export functionality
- 📈 Confidence analysis and statistics

## Installation

### 1. Clone the repository:
```bash
git clone <repository-url>
cd marksheet-extractor
```

### 2. Install system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Download and install Poppler from: https://blog.alivate.com.au/poppler-windows/

### 3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Quick Start

**Option 1: Automatic setup**
```bash
python run.py  # This will install everything and start the app
```

**Option 2: Manual start**
```bash
streamlit run app.py
```

## Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

3. **Enter your OpenAI API key** in the sidebar

4. **Upload a marksheet** (JPG, PNG, or PDF format, max 10MB)

5. **Click "Extract Data"** to process the marksheet

6. **View results** in multiple formats:
   - Formatted view with confidence indicators
   - Raw JSON output with download option
   - Confidence analysis and statistics

## Configuration

### Model Selection
Choose from available OpenAI models:
- `gpt-4-vision-preview` (recommended for images)
- `gpt-4` (text analysis)
- `gpt-3.5-turbo` (faster, less accurate)

### Confidence Threshold
Adjust the minimum confidence threshold to filter out uncertain extractions.

## Data Schema

The extracted data follows this JSON structure:

```json
{
  "candidate_details": {
    "name": {"value": "John Doe", "confidence": 0.95},
    "father_name": {"value": "Robert Doe", "confidence": 0.85},
    "mother_name": {"value": "Jane Doe", "confidence": 0.80},
    "roll_number": {"value": "12345", "confidence": 0.90},
    "registration_number": {"value": "REG123", "confidence": 0.85},
    "date_of_birth": {"value": "1995-06-15", "confidence": 0.88},
    "exam_year": {"value": "2023", "confidence": 0.92},
    "board_university": {"value": "State Board", "confidence": 0.87},
    "institution": {"value": "ABC School", "confidence": 0.83}
  },
  "subjects": [
    {
      "subject_name": {"value": "Mathematics", "confidence": 0.92},
      "max_marks": {"value": 100, "confidence": 0.88},
      "obtained_marks": {"value": 85, "confidence": 0.90},
      "grade": {"value": "A", "confidence": 0.85}
    },
    {
      "subject_name": {"value": "Physics", "confidence": 0.89},
      "max_marks": {"value": 100, "confidence": 0.88},
      "obtained_marks": {"value": 78, "confidence": 0.87},
      "grade": {"value": "B+", "confidence": 0.82}
    }
  ],
  "overall_result": {
    "total_marks": {"value": 425, "confidence": 0.80},
    "percentage": {"value": 85.0, "confidence": 0.82},
    "grade": {"value": "First Class", "confidence": 0.88},
    "division": {"value": "I", "confidence": 0.85},
    "result_status": {"value": "Pass", "confidence": 0.95}
  },
  "document_metadata": {
    "issue_date": {"value": "2023-06-15", "confidence": 0.75},
    "issue_place": {"value": "Mumbai", "confidence": 0.70},
    "document_type": {"value": "Marksheet", "confidence": 0.90}
  }
}
```

## Confidence Scoring

Confidence scores range from 0.0 to 1.0 and are calculated using multiple factors:

- **OCR Confidence (30%)**: Raw OCR confidence from Tesseract
- **Pattern Matching (20%)**: Adherence to expected field patterns
- **Context Relevance (20%)**: Relevance to surrounding text
- **Completeness (10%)**: Field completeness indicators
- **LLM Base Score (20%)**: Initial confidence from language model

### Confidence Ranges:
- **0.9-1.0**: Very high confidence (clear, unambiguous)
- **0.7-0.8**: High confidence (clear with minor issues)
- **0.5-0.6**: Medium confidence (somewhat unclear)
- **0.3-0.4**: Low confidence (unclear or ambiguous)
- **0.1-0.2**: Very low confidence (barely readable)
- **0.0**: No confidence (field not found)

## Approach & Methodology

### 1. OCR Processing
- **Image Preprocessing**: Grayscale conversion, denoising, adaptive thresholding
- **Text Extraction**: Tesseract OCR with optimized configuration
- **PDF Handling**: Convert PDF pages to images for processing

### 2. LLM Integration
- **Structured Prompting**: Clear instructions for JSON schema adherence
- **Model Selection**: Support for multiple OpenAI models
- **Error Handling**: Robust JSON parsing and validation

### 3. Confidence Calculation
Multi-factor scoring system combining:
- OCR reliability scores
- Pattern matching for field types (dates, numbers, grades)
- Contextual relevance analysis
- Completeness indicators

### 4. Data Validation
- Field type validation (dates, numbers, text)
- Pattern matching for specific field types
- Confidence calibration and normalization

## Troubleshooting

### Common Issues:

1. **Tesseract not found**:
   - Make sure Tesseract is installed and in your PATH
   - On Windows, you may need to set the path manually in the code

2. **PDF processing fails**:
   - Ensure Poppler is installed for PDF to image conversion
   - Check that the PDF is not password protected

3. **Low extraction accuracy**:
   - Try using a higher resolution image (300 DPI recommended)
   - Ensure the marksheet is clearly visible and not skewed
   - Use `gpt-4-vision-preview` model for better image understanding

4. **API errors**:
   - Verify your OpenAI API key is correct and has available credits
   - Check your internet connection
   - Ensure you have access to the selected model

5. **Memory issues with large files**:
   - Ensure files are under 10MB
   - For large PDFs, consider splitting into smaller files

## File Structure

```
marksheet-extractor/
├── app.py                    # Main Streamlit application
├── services/
│   ├── __init__.py
│   ├── ocr_service.py       # OCR processing
│   ├── llm_service.py       # LLM integration
│   └── confidence_service.py # Confidence scoring
├── utils/
│   ├── __init__.py
│   └── file_utils.py        # File handling utilities
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── .env                    # Your environment variables (create this)
├── run.py                  # Quick start script
├── temp/                   # Temporary files (auto-created)
├── test_marksheets/        # Sample marksheets (optional)
└── README.md              # This file
```

## Testing

### Sample Test Cases
1. Upload a clear, high-resolution marksheet image
2. Test with a scanned PDF marksheet
3. Try with images at different orientations
4. Test with partially damaged/unclear marksheets

### Expected Results
- Clear marksheets should achieve >0.8 overall confidence
- All major fields should be extracted
- JSON output should be valid and complete

## API Integration (Optional)

While this is a Streamlit app, you can easily convert it to a FastAPI backend:

1. Extract the processing logic from `app.py`
2. Create FastAPI endpoints
3. Return JSON responses
4. Add authentication if needed

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify your OpenAI API key and credits
4. Create an issue on GitHub with detailed error information

## Acknowledgments

- OpenAI for the GPT models
- Tesseract OCR for text extraction
- Streamlit for the web interface
- pdf2image for PDF processing