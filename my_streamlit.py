import streamlit as st
from PIL import Image
import cv2
import requests
import pytesseract
import numpy as np
import medspacy
import spacy
import en_ner_bc5cdr_md
import en_core_med7_trf
nlp = spacy.load("en_ner_bc5cdr_md")
nlp = spacy.load("en_core_med7_trf")
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import re
import streamlit_app

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
language_models = {
    "Med7": "en_core_med7_trf",
    "bc5cdr": "en_ner_bc5cdr_md",
    "custom": "model-best"
}
selected_model = st.selectbox("Select Language Model", list(language_models.keys()))

# HTML markup for the thumbnail-sized image
image_html = """
    <div class="footer">
    <p>Powered By : </p>
        <img src="https://play-lh.googleusercontent.com/hPbxJhnceggrITJpS-iqh-UDVqPWgWKIsQGBPL9AFnjSfKi9jm_VpOIvijhbupeO5A" alt="Spacy Logo">
    <img src="data:image/png;base64,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" alt="Spacy Logo">
        
    </div>
"""

# Placeholder function for NER model
def perform_ner(text):
    nlp =  spacy.load(language_models[selected_model])
    doc = nlp(text)
    medications = []
    for ent in doc.ents:
        
        medications.append(ent.text)
    
    return medications

def preprocess_text(text):
    # Load SpaCy pipeline
    nlp2 = spacy.load("en_core_web_sm")

    # Tokenization, lowercasing, and lemmatization
    doc = nlp2(text)
    tokens = [token.lemma_.strip() for token in doc]
    # Remove stop words and non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha() and token not in STOP_WORDS]

    # Remove punctuation and special characters
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]

    # Remove extra whitespace
    tokens = [token.strip() for token in tokens if token.strip()]

    # Join tokens back into a cleaned text
    cleaned_text = ' '.join(tokens)

    return cleaned_text

def get_drug_information(drug_name):
    # Replace this with the actual endpoint of your drug information API
    
    base_url="https://www.drugs.com/"
    sub_urls=["","mtm/","dosage/","cons/"]
    for url in sub_urls:
        try:
            complete_url = base_url + url + "{name}.html".format(name=drug_name)
            response = requests.get(complete_url)
            if response.status_code == 200:
                return complete_url
        except requests.exceptions.RequestException:
            pass
    return None
    headers = {
	"X-RapidAPI-Key": "5919b6c652msh5ecafc2bb4bf982p116b12jsn0c46f7cfc284",
	"X-RapidAPI-Host": "drug-info-and-price-history.p.rapidapi.com"
}
    
    
    # Make a request to the API
    
    
    

# Main Streamlit app code
def main():
    st.title("Medical Prescription NER")
    st.write("Upload an image of a medical prescription to extract entities.")
    image = Image.open(r"C:\Users\ppava\OneDrive\Desktop\fyp\final_fyp\test-ner\1.jpg")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    df = pd.DataFrame(columns=['Drug', 'Info'])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform NER on the image
        cleaned_text=preprocess_text(text)
        #st.write(cleaned_text)
        entities = perform_ner(cleaned_text)
        
        # Display the extracted entities
        st.subheader("Extracted Entities:")
        
        for entity in entities: 
            st.write(entity)
            with st.spinner("Fetching drug information..."):
                # Get drug information from the API
                drug_info = get_drug_information(entity)
            

            if drug_info:
                # Display drug information in a table format
                df = pd.concat([df,pd.DataFrame({'Drug': [entity], 'Info': [drug_info]})], ignore_index=True)
        
    st.table(df)
    st.markdown('### Links')
    for _, row in df.iterrows():
        if row['Info']:
            st.markdown(f"[{row['Drug']}]({row['Info']})", unsafe_allow_html=True)
   

# Run the app
if __name__ == '__main__':
    main()
