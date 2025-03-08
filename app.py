import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
from fpdf import FPDF  # For PDF generation
import io  # To handle in-memory PDF generation
import gdown  # For downloading model from Google Drive
import os

# Google Drive Model Link (Replace with your file ID)
GOOGLE_DRIVE_FILE_ID = "1Z1t83s-7hAgAFw2yVFYfnsKeW2KKxtGu"
MODEL_PATH = r"C:\Users\rekha\Downloads\combinex\combinexxy_cnn.h5"

# Function to download model from Google Drive
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model...")
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = download_and_load_model()

# Title & Description
st.title("🩺 Glaucoma Detection App")
st.write("Upload an eye image, and the model will predict whether glaucoma is detected or not.")

uploaded_file = st.file_uploader("📤 Upload an Eye Image", type=["jpg", "png", "jpeg"])

# Session state for storing past results
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file:
    try:
        # Image Processing
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))
        st.image(img, caption="📷 Uploaded Image", use_column_width=True)

        # Normalize & Expand Dimensions
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        if model:
            with st.spinner("🔍 Analyzing..."):
                prediction = model.predict(img_array)[0][0]
                confidence = round(prediction * 100, 2)

            result = "🛑 **Glaucoma Detected**" if confidence > 50 else "✅ **Healthy Eye**"
            st.write(f"### Prediction: {result}")
            st.write(f"**Confidence Score:** {confidence}%")

            # Store result in session state
            st.session_state.history.append({"Image": uploaded_file.name, "Result": result, "Confidence": confidence})

            # **1️⃣ Confidence Graph using Matplotlib**
            fig, ax = plt.subplots()
            ax.bar(["Healthy Eye", "Glaucoma"], [100 - confidence, confidence], color=['green', 'red'])
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Confidence Level")
            st.pyplot(fig)

            # **2️⃣ Generate PDF Report**
            def generate_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(200, 10, "Glaucoma Detection Report", ln=True, align="C")

                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, f"Image Name: {uploaded_file.name}", ln=True)
                pdf.cell(200, 10, f"Prediction: {result}", ln=True)
                pdf.cell(200, 10, f"Confidence Score: {confidence}%", ln=True)

                pdf_output = io.BytesIO()
                pdf.output(pdf_output)
                return pdf_output.getvalue()

            pdf_data = generate_pdf()
            st.download_button(label="📥 Download Report", data=pdf_data, file_name="glaucoma_report.pdf", mime="application/pdf")

        else:
            st.error("Model is not available. Please check the file path.")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

# **3️⃣ Display Past Predictions**
if st.session_state.history:
    st.write("### 📌 Previous Predictions")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
