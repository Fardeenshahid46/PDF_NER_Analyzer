import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
import pandas as pd
from itertools import islice

st.set_page_config(page_title="PDF NER Analyzer", layout="wide")

st.title("📄 PDF Named Entity Recognition (NER) Analyzer")
st.write("Upload any PDF (text-based or scanned) to extract entities.")

# ------------------------
# Upload PDF
# ------------------------
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# ------------------------
# Load NER Model (cached)
# ------------------------
@st.cache_resource
def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipeline

ner_pipeline = load_ner_model()

# ------------------------
# Function: Split text into chunks
# ------------------------
def chunk_text(text, size=500):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

# ------------------------
# Process PDF
# ------------------------
if uploaded_file:
    st.info("Processing PDF... This may take a few seconds.")
    text = ""
    
    # Try text extraction first
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except:
        st.warning("Failed to extract text with pdfplumber.")
    
    # If no text, use OCR
    if len(text.strip()) == 0:
        st.info("No text found! Using OCR for scanned PDF...")
        images = convert_from_bytes(uploaded_file.read())
        for img in images:
            text += pytesseract.image_to_string(img)
    
    if len(text.strip()) == 0:
        st.error("⚠️ Could not extract any text from PDF.")
    else:
        st.success("✅ Text extracted successfully!")
        
        # ------------------------
        # Chunked NER processing with progress
        # ------------------------
        st.info("Running Named Entity Recognition...")
        all_entities = []
        chunks = list(chunk_text(text, size=500))
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            entities = ner_pipeline(chunk)
            all_entities.extend(entities)
            progress_bar.progress((i+1)/len(chunks))
        
        if len(all_entities) == 0:
            st.warning("No named entities found in the PDF.")
        else:
            # Convert to DataFrame
            df = pd.DataFrame(all_entities)
            df = df[["word", "entity_group", "score"]].rename(columns={
                "word": "Entity", "entity_group": "Type", "score": "Confidence"
            })
            df["Confidence"] = df["Confidence"].apply(lambda x: round(x, 3))
            
            # ------------------------
            # Streamlit Tabs
            # ------------------------
            tab1, tab2, tab3 = st.tabs(["📄 Text Preview", "📝 Entities Table", "📊 Charts"])
            
            with tab1:
                st.subheader("PDF Text Preview")
                st.text_area("Extracted Text", text[:10000]+"..." if len(text)>10000 else text, height=300)
            
            with tab2:
                st.subheader("Extracted Named Entities")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Entities as CSV", data=csv, file_name="entities.csv", mime="text/csv")
            
            with tab3:
                st.subheader("Entity Counts")
                entity_count = df['Type'].value_counts().reset_index()
                entity_count.columns = ['Entity Type', 'Count']
                st.bar_chart(entity_count.set_index('Entity Type'))
                
                st.subheader("Entity Metrics")
                for etype, count in entity_count.values:
                    st.metric(label=etype, value=int(count))
