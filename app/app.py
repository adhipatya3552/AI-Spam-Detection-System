import sys
import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import tempfile

# Add the project root to sys.path to find the 'src' module
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
if os.path.join(base_dir, "src") not in sys.path:
    sys.path.insert(0, os.path.join(base_dir, "src"))

# =========================================
# AUTO-TRAINING FALLBACK (Streamlit Cloud support)
# =========================================
model_path = os.path.join(base_dir, "models", "spam_pipeline.pkl")
if not os.path.exists(model_path):
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    with st.spinner("🤖 First-time setup: Training the machine learning model on the public dataset..."):
        from src.train_model import train
        train()

# Import prediction functions after model ensures existence
from src.predict import predict_message, extract_email_text

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Spam Detection System",
    page_icon="📩",
    layout="centered"
)

# Load model
@st.cache_resource
def get_model():
    return joblib.load(model_path)

model = get_model()

# =========================================
# TITLE
# =========================================
st.title("📩 AI-Powered Spam & Threat Detection")
st.write("Analyze messages and detect spam using NLP and Machine Learning.")

# =========================================
# TABS CREATION
# =========================================
tab1, tab2 = st.tabs(["⚡ Scan & Upload", "📊 Model Evaluation Dashboard"])

# =========================================
# TAB 1: SCAN & UPLOAD
# =========================================
with tab1:
    st.subheader("📊 Dataset Distribution")

    df_path = os.path.join(base_dir, "data", "spam.csv")
    df = pd.read_csv(df_path, encoding='latin-1')
    chart_data = df['label'].value_counts()
    st.bar_chart(chart_data)

    st.subheader("⚡ Live Email Scanning")
    message = st.text_area("Enter Email Message", height=150)

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

    st.subheader("📂 Upload Email File")
    uploaded_file = st.file_uploader("Upload a .txt email file", type=["txt"])

    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        cleaned_content = extract_email_text(file_content)

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
            st.text_area("Uploaded Email Content (Cleaned)", cleaned_content, height=150)
        else:
            st.text_area("Raw Email Code Content", file_content, height=150)

        if st.button("Scan Uploaded File"):
            with st.spinner("Scanning Uploaded File..."):
                prediction, confidence = predict_message(cleaned_content)

            if prediction == 1:
                st.error("🚨 Spam Email Detected")
                st.write(f"Threat Level: {confidence:.2f}%")
                st.progress(int(confidence))

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

# =========================================
# TAB 2: MODEL EVALUATION DASHBOARD
# =========================================
with tab2:
    st.subheader("📈 Model Evaluation & Performance Metrics")
    st.write(
        "Evaluate the performance of the trained ML pipeline on a 20% test split "
        "of the SMS Spam Collection dataset."
    )

    # Cache computations to avoid laggy performance
    @st.cache_data
    def get_evaluation_data():
        from src.data_loader import load_data
        from src.preprocess import clean_text
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

        data_path = os.path.join(base_dir, "data", "spam.csv")
        df = load_data(data_path)
        df['text'] = df['text'].apply(clean_text)

        X = df['text']
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_path = os.path.join(base_dir, "models", "spam_pipeline.pkl")
        model = joblib.load(model_path)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions)
        }

        class_report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions).tolist()

        return metrics, class_report, cm

    # Button to trigger evaluation
    if st.button("Run Model Evaluation"):
        with st.spinner("Computing evaluation metrics..."):
            metrics, class_report, cm = get_evaluation_data()

        # Display Metrics in Columns
        st.markdown("### Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("Recall", f"{metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{metrics['f1']:.2%}")

        st.write("---")
        c1, c2 = st.columns([1.2, 1.0])

        with c1:
            st.markdown("### Confusion Matrix")
            # Generate the plot
            import numpy as np
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                np.array(cm),
                annot=True,
                fmt='d',
                cmap='Reds',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig)

        with c2:
            st.markdown("### Classification Report")
            st.text(class_report)

        st.write("---")
        st.markdown("### 📄 Export Report")
        st.write("Export these evaluation results and visual metrics as a downloadable PDF report.")

        # Create temporary file for Confusion Matrix image to embed in PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            fig.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            tmp_img_path = tmp_file.name

        try:
            # Generate PDF
            from fpdf import FPDF
            from fpdf.enums import XPos, YPos

            class EvaluationReportPDF(FPDF):
                def header(self):
                    self.set_font('Helvetica', 'B', 16)
                    self.cell(0, 10, 'AI Spam Detection System', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
                    self.set_font('Helvetica', 'I', 11)
                    self.cell(0, 10, 'Model Evaluation & Performance Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
                    self.ln(10)

                def footer(self):
                    self.set_y(-15)
                    self.set_font('Helvetica', 'I', 8)
                    self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

            def generate_pdf_report(metrics, class_report, cm_img_path):
                pdf = EvaluationReportPDF()
                pdf.alias_nb_pages()
                pdf.add_page()
                
                # Title & Metadata
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 8, 'Report Metadata', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font('Helvetica', '', 10)
                pdf.cell(0, 6, f'Generated On: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.cell(0, 6, 'Base Dataset: SMS Spam Collection (Kaggle)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.cell(0, 6, 'Model Type: Trained Classifier (Naive Bayes / Logistic Regression)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(5)

                # Key Metrics Table
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 8, 'Key Performance Metrics', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
                # Header
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(50, 8, 'Metric', border=1, align='C')
                pdf.cell(50, 8, 'Score', border=1, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
                # Rows
                pdf.set_font('Helvetica', '', 10)
                for name, val in [("Accuracy", metrics["accuracy"]), 
                                  ("Precision", metrics["precision"]), 
                                  ("Recall", metrics["recall"]), 
                                  ("F1 Score", metrics["f1"])]:
                    pdf.cell(50, 8, name, border=1)
                    pdf.cell(50, 8, f'{val:.2%}', border=1, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
                pdf.ln(8)

                # Classification Report
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 8, 'Detailed Classification Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font('Courier', '', 9)
                for line in class_report.split('\n'):
                    pdf.cell(0, 4, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    
                pdf.ln(8)

                # Confusion Matrix Image
                if os.path.exists(cm_img_path):
                    pdf.set_font('Helvetica', 'B', 12)
                    pdf.cell(0, 8, 'Confusion Matrix Visualization', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.image(cm_img_path, x=45, y=pdf.get_y(), w=110)
                    
                return pdf.output()

            pdf_data = generate_pdf_report(metrics, class_report, tmp_img_path)
            
            # Ensure correct format for download
            if isinstance(pdf_data, bytearray):
                pdf_bytes = bytes(pdf_data)
            elif isinstance(pdf_data, str):
                pdf_bytes = pdf_data.encode('latin-1')
            else:
                pdf_bytes = pdf_data

            # Download Button
            st.download_button(
                label="📥 Download Evaluation Report (PDF)",
                data=pdf_bytes,
                file_name=f"spam_detection_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error generating PDF report: {e}")

        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_img_path)
            except Exception:
                pass