import streamlit as st
from rapidocr_onnxruntime import RapidOCR
import tempfile
import os
from typing import List, Tuple
from dataclasses import dataclass

# OCRProcessor class
@dataclass
class Detection:
    coords: List[Tuple[float, float]]
    text: str
    conf: float
    @property
    def y(self): return self.coords[0][1]
    @property
    def x(self): return self.coords[0][0]

class OCRProcessor:
    def __init__(self, img_path: str):
        self.engine = RapidOCR()
        self.img_path = img_path
        self.results = None

    def process(self):
        self.results = [Detection(det[0], det[1], det[2]) for det in self.engine(self.img_path)[0]]

    def group_lines(self, y_thresh=30):
        if not self.results: raise ValueError("No results")
        lines, curr, y = [], [], self.results[0].y
        for d in sorted(self.results, key=lambda x: x.y):
            if abs(d.y - y) <= y_thresh: curr.append(d)
            else:
                lines.append(sorted(curr, key=lambda x: x.x))
                curr, y = [d], d.y
        if curr: lines.append(sorted(curr, key=lambda x: x.x))
        return lines

st.set_page_config(
    page_title="OCR Text Extractor",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""<h7>Demo by <a href="https://www.linkedin.com/in/almajdoub/">Almjdoub</a></h7>""", unsafe_allow_html=True)

st.title("OCR text extractor")
st.markdown("Image to Text extractor utilizing OCR (Optical Character Recognition) for English characters")




# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

if uploaded_file is not None:
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.'+uploaded_file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    # Display the uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    # Process the image
    try:
        processor = OCRProcessor(temp_path)
        processor.process()
        
        # Get grouped lines
        lines = processor.group_lines()
        
        with col2:
            st.subheader("Extracted Text")
            # Display extracted text
            extracted_text = '\n'.join(' '.join(d.text for d in line) for line in lines)
            st.text_area("", extracted_text, height=300)
            
            # Add download button
            st.download_button(
                label="Download extracted text",
                data=extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
    finally:
        # Clean up temporary file
        os.unlink(temp_path)
st.markdown("""
---
### Instructions:
1. Upload an image containing text (.png, .jpg, or .jpeg)
2. Wait for the app to process the image
3. Copy or download the extracted text

*note: this is for experimental purposes only, confidence % may vary*

""") 
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextArea>div>div>textarea {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""Powered by [RapidOCR](https://github.com/RapidAI/RapidOCR)""")