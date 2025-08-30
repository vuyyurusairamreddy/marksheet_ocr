import streamlit as st
import os
import json
import time
from datetime import datetime
import logging
from typing import Optional
from PIL import Image
import pandas as pd

from config import Config
from services.ocr_service import OCRService
from services.llm_service import LLMService
from services.confidence_service import ConfidenceService
from utils.file_utils import FileUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Marksheet Extraction API",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
@st.cache_resource
def initialize_services():
    return {
        'ocr': OCRService(),
        'llm': LLMService(), # type: ignore
        'confidence': ConfidenceService(),
        'file_utils': FileUtils()
    }

services = initialize_services()

# Create upload directory
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

def main():
    st.title("🎓 Marksheet Extraction API")
    st.markdown("**AI-powered marksheet data extraction with confidence scoring**")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a marksheet (JPG/PNG/PDF)
        2. Wait for AI processing
        3. View extracted data with confidence scores
        
        **Supported formats:**
        - Images: JPG, JPEG, PNG
        - Documents: PDF
        - Max size: 10 MB
        """)
        
        st.header("⚙️ Settings")
        api_key = st.text_input("API Key (Optional)", type="password", help="Enter API key if required")
        show_raw_text = st.checkbox("Show Raw OCR Text", help="Display extracted text before processing")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.1)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose a marksheet file",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="Upload a marksheet image or PDF file (max 10MB)"
        )
        
        if uploaded_file is not None:
            # File validation
            file_size = len(uploaded_file.getvalue())
            if file_size > Config.MAX_FILE_SIZE:
                st.error(f"File size ({file_size/(1024*1024):.1f} MB) exceeds maximum limit ({Config.MAX_FILE_SIZE/(1024*1024)} MB)")
                return
            
            # Display file info
            st.success(f"File uploaded: {uploaded_file.name} ({file_size/(1024*1024):.1f} MB)")
            
            # Show image preview if it's an image
            if uploaded_file.type.startswith('image/'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("🤖 Processing")
        
        if uploaded_file is not None:
            if st.button("🚀 Extract Data", type="primary"):
                process_file(uploaded_file, show_raw_text, confidence_threshold, api_key)
        else:
            st.info("Please upload a file to begin extraction")

def process_file(uploaded_file, show_raw_text, confidence_threshold, api_key=None):
    """Process the uploaded file and extract marksheet data"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        start_time = time.time()
        
        # Step 1: Save uploaded file
        status_text.text("💾 Saving uploaded file...")
        progress_bar.progress(10)
        
        temp_file_path = services['file_utils'].save_uploaded_file(uploaded_file)
        
        # Step 2: Extract text using OCR
        status_text.text("👁️ Extracting text with OCR...")
        progress_bar.progress(30)
        
        if uploaded_file.type == 'application/pdf':
            raw_text = services['ocr'].extract_from_pdf(temp_file_path)
        else:
            raw_text = services['ocr'].extract_from_image(temp_file_path)
        
        if show_raw_text:
            st.subheader("📄 Raw OCR Text")
            st.text_area("Extracted Text", raw_text, height=200)
        
        # Step 3: Process with LLM
        status_text.text("🧠 Processing with AI...")
        progress_bar.progress(60)
        
        structured_data = services['llm'].extract_structured_data(raw_text)
        
        # Step 4: Calculate confidence scores
        status_text.text("📊 Calculating confidence scores...")
        progress_bar.progress(80)
        
        final_data = services['confidence'].add_confidence_scores(structured_data, raw_text)
        
        # Step 5: Complete
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        processing_time = time.time() - start_time
        
        # Display results
        display_results(final_data, processing_time, confidence_threshold)
        
        # Cleanup
        services['file_utils'].cleanup_file(temp_file_path)
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        logger.error(f"Processing error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def display_results(data, processing_time, confidence_threshold):
    """Display the extracted results in a user-friendly format"""
    
    st.subheader("📊 Extraction Results")
    
    # Processing info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    with col2:
        st.metric("Fields Extracted", len(flatten_dict(data.get('extracted_data', {}))))
    with col3:
        avg_confidence = calculate_average_confidence(data.get('extracted_data', {}))
        st.metric("Average Confidence", f"{avg_confidence:.2f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "👤 Candidate Info", "📚 Subjects", "📄 Raw JSON"])
    
    with tab1:
        display_summary(data, confidence_threshold)
    
    with tab2:
        display_candidate_info(data.get('extracted_data', {}).get('candidate_details', {}), confidence_threshold)
    
    with tab3:
        display_subjects(data.get('extracted_data', {}).get('subjects', []), confidence_threshold)
    
    with tab4:
        st.json(data)
        
        # Download button for JSON
        json_str = json.dumps(data, indent=2)
        st.download_button(
            label="💾 Download JSON",
            data=json_str,
            file_name=f"marksheet_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def display_summary(data, confidence_threshold):
    """Display summary information"""
    extracted_data = data.get('extracted_data', {})
    
    # High-level stats
    st.subheader("📈 Extraction Summary")
    
    candidate = extracted_data.get('candidate_details', {})
    result_info = extracted_data.get('result_info', {})
    
    # Key information cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Student Name:** {candidate.get('name', {}).get('value', 'Not found')}
        **Roll Number:** {candidate.get('roll_number', {}).get('value', 'Not found')}
        **Board/University:** {candidate.get('board_university', {}).get('value', 'Not found')}
        """)
    
    with col2:
        st.info(f"""
        **Overall Result:** {result_info.get('overall_grade', {}).get('value', 'Not found')}
        **Exam Year:** {candidate.get('exam_year', {}).get('value', 'Not found')}
        **Total Subjects:** {len(extracted_data.get('subjects', []))}
        """)
    
    # Confidence analysis
    st.subheader("🎯 Confidence Analysis")
    confidence_stats = analyze_confidence(extracted_data, confidence_threshold)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Confidence", f"{confidence_stats['high']} fields", delta=f"≥{Config.HIGH_CONFIDENCE_THRESHOLD}")
    with col2:
        st.metric("Medium Confidence", f"{confidence_stats['medium']} fields", delta=f"{confidence_threshold}-{Config.HIGH_CONFIDENCE_THRESHOLD}")
    with col3:
        st.metric("Low Confidence", f"{confidence_stats['low']} fields", delta=f"<{confidence_threshold}", delta_color="inverse")

def display_candidate_info(candidate_details, confidence_threshold):
    """Display candidate information with confidence scores"""
    st.subheader("👤 Candidate Details")
    
    if not candidate_details:
        st.warning("No candidate details found")
        return
    
    for field, data in candidate_details.items():
        if isinstance(data, dict) and 'value' in data:
            confidence = data.get('confidence', 0)
            confidence_color = get_confidence_color(confidence, confidence_threshold)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{field.replace('_', ' ').title()}:** {data['value']}")
            with col2:
                st.markdown(f"<span style='color: {confidence_color}'>●</span> {confidence:.2f}", unsafe_allow_html=True)

def display_subjects(subjects, confidence_threshold):
    """Display subjects information in a table format"""
    st.subheader("📚 Subject-wise Marks")
    
    if not subjects:
        st.warning("No subjects found")
        return
    
    # Convert to DataFrame for better display
    rows = []
    for subject in subjects:
        if isinstance(subject, dict):
            row = {}
            for key, value in subject.items():
                if isinstance(value, dict) and 'value' in value:
                    row[key.replace('_', ' ').title()] = value['value']
                    row[f"{key}_confidence"] = value.get('confidence', 0)
                else:
                    row[key.replace('_', ' ').title()] = str(value)
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No valid subject data found")

def flatten_dict(d, prefix=''):
    """Flatten nested dictionary for counting fields"""
    items = []
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if 'value' in value:
                items.append(new_key)
            else:
                items.extend(flatten_dict(value, new_key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]"))
    return items

def calculate_average_confidence(data):
    """Calculate average confidence score"""
    confidences = []
    
    def extract_confidences(obj):
        if isinstance(obj, dict):
            if 'confidence' in obj:
                confidences.append(obj['confidence'])
            for value in obj.values():
                extract_confidences(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_confidences(item)
    
    extract_confidences(data)
    return sum(confidences) / len(confidences) if confidences else 0

def analyze_confidence(data, threshold):
    """Analyze confidence distribution"""
    confidences = []
    
    def extract_confidences(obj):
        if isinstance(obj, dict):
            if 'confidence' in obj:
                confidences.append(obj['confidence'])
            for value in obj.values():
                extract_confidences(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_confidences(item)
    
    extract_confidences(data)
    
    high = sum(1 for c in confidences if c >= Config.HIGH_CONFIDENCE_THRESHOLD)
    medium = sum(1 for c in confidences if threshold <= c < Config.HIGH_CONFIDENCE_THRESHOLD)
    low = sum(1 for c in confidences if c < threshold)
    
    return {'high': high, 'medium': medium, 'low': low}

def get_confidence_color(confidence, threshold):
    """Get color based on confidence level"""
    if confidence >= Config.HIGH_CONFIDENCE_THRESHOLD:
        return "green"
    elif confidence >= threshold:
        return "orange"
    else:
        return "red"

if __name__ == "__main__":
    main()