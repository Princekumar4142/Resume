import streamlit as st
import pdfplumber
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==============================
# NLTK Setup (first time only)
# ==============================
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ==============================
# Load Saved Model Files
# ==============================
model = joblib.load("resume_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==============================
# Functions
# ==============================

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(words)


def predict_resume_category(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    cleaned = preprocess_text(text)

    features = tfidf.transform([cleaned]).toarray()

    prediction = model.predict(features)
    category = label_encoder.inverse_transform(prediction)[0]

    prob = model.predict_proba(features)
    confidence = max(prob[0]) * 100

    return category, confidence


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get category prediction instantly 🚀")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Processing... ⏳")

    category, confidence = predict_resume_category("temp_resume.pdf")

    st.success(f"🎯 Predicted Category: {category}")
    st.info(f"📊 Confidence: {confidence:.2f}%")
