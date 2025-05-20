import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load NLTK stuff
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load saved model and vectorizer
with open('phishing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

# Prediction function
def predict_email(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    return "📛 Phishing Email" if prediction == 1 else "✅ Legitimate Email"

# Streamlit UI
st.set_page_config(page_title="PhishFence", layout="centered")
st.title("🛡️ PhishFence: Phishing Email Detector")
st.markdown("Enter suspicious email content below to check if it's phishing or legitimate.")

email_input = st.text_area("✉️ Email Content", height=250)

if st.button("🔍 Analyze"):
    if email_input.strip():
        result = predict_email(email_input)
        st.success(f"Result: **{result}**")
    else:
        st.warning("Please enter some email content.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Scikit-learn")
