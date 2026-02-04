import os
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re

# -----------------------------
# Silence TensorFlow logs
# -----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection App")
st.write("Enter a news article below and check whether it is **REAL** or **FAKE**.")

# -----------------------------
# Constants (must match training)
# -----------------------------
MAX_LEN = 300

# -----------------------------
# Text cleaning (same as training)
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# -----------------------------
# Load tokenizer (SAFE to cache)
# -----------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()

# -----------------------------
# Load TFLite model (Python 3.13 safe)
# -----------------------------
def load_tflite_model():
    interpreter = tf.lite.Interpreter(
        model_path="fake_news_model.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# -----------------------------
# Prediction function
# -----------------------------
def predict_news(text):
    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAX_LEN, padding="post"
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        padded.astype(np.int32)
    )
    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )[0][0]

    return prediction

# -----------------------------
# UI
# -----------------------------
user_input = st.text_area(
    "📝 Paste news text here:",
    height=200,
    placeholder="Example: Government announces new economic reforms..."
)

if st.button("🔍 Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            score = predict_news(user_input)

        if score > 0.5:
            st.error(f"🚨 FAKE NEWS\n\nConfidence: {score:.2f}")
        else:
            st.success(f"✅ REAL NEWS\n\nConfidence: {1 - score:.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow Lite & Streamlit")
