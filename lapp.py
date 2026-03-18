import os
import streamlit as st
import pickle

# Absolute path fix
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")


# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Language Detector", page_icon="🌍")

# Custom CSS
st.markdown("""
    <style>
    /* Full page background */
    .stApp {
        background: linear-gradient(to right, #eef2f3, #dfe9f3);
    }

    /* Main Title */
    .main-title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }

    /* Text area label */
    .input-label {
        font-size: 22px;
        font-weight: bold;
        color: #1f4e79;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50, #2ecc71);
        color: white;
        border-radius: 12px;
        height: 3.2em;
        width: 220px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #43a047, #27ae60);
    }
    div[data-baseweb="notification"] {
        background: linear-gradient(135deg, #232526, #414345) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 15px !important;
    </style>
""", unsafe_allow_html=True)

# Big Title
st.markdown('<div class="main-title">🌍 Language Detection App</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌍 Supported Languages")
for lang in model.classes_:
    st.sidebar.markdown(f"• {lang}")

# Bold & Styled Label
st.markdown('<div class="input-label">✍️ Enter your text here</div>', unsafe_allow_html=True)

# Input box
user_input = st.text_area("")

# Button
if st.button("Detect Language"):
    if user_input:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data).max()

        if probability < 0.6:
            st.warning("⚠️ Please enter text in supported languages only!")
        else:
            st.success(f"✅ Predicted Language: {prediction}")
    else:
        st.warning("⚠️ Please enter some text first!")