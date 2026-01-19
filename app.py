import streamlit as st
import joblib
import os
import nltk

# Load the model
model_path = 'sms_spam_app/models/spam_classifier.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Page Config
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #ced4da;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .ham {
        background-color: #28a745;
    }
    .spam {
        background-color: #dc3545;
    }
    h1 {
        color: #333333;
        text-align: center;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .description {
        text-align: center;
        color: #666666;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Verify if an SMS message is Spam or Safe (Ham) instantly.</p>", unsafe_allow_html=True)

# Input
sms_input = st.text_area("Enter the SMS message below:", height=150, placeholder="Type or paste the message here...")

# Prediction logic
if st.button("Check Message"):
    if sms_input.strip() == "":
        st.warning("Please enter a message to check.")
    else:
        if model:
            with st.spinner("Analyzing message..."):
                prediction = model.predict([sms_input])[0]
                
                if prediction == 'spam':
                    st.markdown(
                        """<div class="result-card spam">🚨 SPAM DETECTED</div>""", 
                        unsafe_allow_html=True
                    )
                    st.error("This message looks like a potential spam or scam.")
                else:
                    st.markdown(
                        """<div class="result-card ham">✅ NOT SPAM (HAM)</div>""", 
                        unsafe_allow_html=True
                    )
                    st.success("This message appears to be safe.")
        else:
            st.error("Model file not found. Please ensure the model is trained.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>Built with Streamlit & Scikit-learn</div>", unsafe_allow_html=True)
