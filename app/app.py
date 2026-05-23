import sys
import os

# Add the project root to sys.path to find the 'src' module
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import streamlit as st
import pandas as pd
import joblib

from src.predict import predict_message, extract_email_text

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Spam Detection System",
    page_icon="📩",
    layout="centered"
)

# =========================================
# LOAD MODEL
# =========================================

model_path = os.path.join(base_dir, "models", "spam_pipeline.pkl")
model = joblib.load(model_path)

# =========================================
# TITLE
# =========================================

st.title("📩 AI-Powered Spam & Threat Detection")

st.write(
    "Analyze messages and detect spam using NLP and Machine Learning."
)

# =========================================
# DATASET CHART
# =========================================

st.subheader("📊 Dataset Distribution")

df_path = os.path.join(base_dir, "data", "spam.csv")
df = pd.read_csv(
    df_path,
    encoding='latin-1'
)

chart_data = df['label'].value_counts()

st.bar_chart(chart_data)

# =========================================
# LIVE MESSAGE SCAN
# =========================================

st.subheader("⚡ Live Email Scanning")

message = st.text_area(
    "Enter Email Message",
    height=200
)

if message:

    with st.spinner("Scanning Email..."):

        prediction, confidence = predict_message(message)

    st.subheader("🔍 Scan Result")

    if prediction == 1:

        st.error("🚨 Spam Message Detected")

        st.write(f"Threat Level: {confidence:.2f}%")

        st.progress(int(confidence))

        # Risk Level
        if confidence > 85:

            st.error("HIGH RISK")

        elif confidence > 60:

            st.warning("MEDIUM RISK")

        else:

            st.success("LOW RISK")

    else:

        st.success("✅ Safe Message")

        st.write(f"Safety Confidence: {confidence:.2f}%")

        st.progress(int(confidence))

# =========================================
# FILE UPLOAD SECTION
# =========================================

st.subheader("📂 Upload Email File")

uploaded_file = st.file_uploader(
    "Upload a .txt email file",
    type=["txt"]
)

if uploaded_file is not None:

    file_content = uploaded_file.read().decode("utf-8")

    # Clean the email content dynamically
    cleaned_content = extract_email_text(file_content)

    # Let the user view raw or cleaned preview if headers are found
    if cleaned_content != file_content:
        view_mode = st.radio(
            "View Mode",
            ["Cleaned Preview", "Raw Email Code"],
            horizontal=True,
            key="file_view_mode"
        )
    else:
        view_mode = "Cleaned Preview"

    if view_mode == "Cleaned Preview":
        st.text_area(
            "Uploaded Email Content (Cleaned)",
            cleaned_content,
            height=200
        )
    else:
        st.text_area(
            "Raw Email Code Content",
            file_content,
            height=200
        )

    if st.button("Scan Uploaded File"):

        with st.spinner("Scanning Uploaded File..."):

            prediction, confidence = predict_message(cleaned_content)

        if prediction == 1:

            st.error("🚨 Spam Email Detected")

            st.write(f"Threat Level: {confidence:.2f}%")

            st.progress(int(confidence))

            # Risk Level
            if confidence > 85:

                st.error("HIGH RISK")

            elif confidence > 60:

                st.warning("MEDIUM RISK")

            else:

                st.success("LOW RISK")

        else:

            st.success("✅ Safe Email")

            st.write(f"Safety Confidence: {confidence:.2f}%")

            st.progress(int(confidence))