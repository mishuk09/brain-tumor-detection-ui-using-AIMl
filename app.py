import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
import cv2

# Load the model, scaler, and label encoder
loaded_objects = joblib.load("model.pkl")
ensemble_model = loaded_objects["model"]
scaler = loaded_objects["scaler"]
label_encoder = loaded_objects["label_encoder"]

# Adding custom HTML and CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #212121;
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 30px;
    }
  
    .upload-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 50px;
        background-color: #fff;
    }
    .stFileUploader {
        border: 2px solid #3498db;
        padding: 20px;
        font-size: 15px;
        border-radius: 10px;
        background-color: #ecf0f1;
    }
    .result {
        font-size: 1.5em;
        color: #34495e;
        text-align: center;
        margin-top: 20px;
        border: 2px solid #3498db;
        padding: 10px;
        border-radius: 10px;
    }
    .result-type{
    font-weight:700;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI for uploading an image
st.markdown('<div class="header">🧠 Brain Tumor Detection App</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to a NumPy array
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Check if the image is already in grayscale format (1 channel)
    if len(img.shape) == 2:  # Single-channel image (grayscale)
        gray_img = img  # No need to convert
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale if it's a color image

    # Preprocess the image (resize, extract features, etc.)
    mean_intensity = np.mean(gray_img)
    texture = np.var(gray_img)  # Example texture feature
    edge_count = np.sum(cv2.Canny(gray_img, 100, 200))  # Example edge feature

    # Feature transformations (as done during training)
    intensity_square = mean_intensity ** 2
    texture_log = np.log1p(texture)

    # Prepare the feature array
    X_new = np.array([[mean_intensity, texture, edge_count, intensity_square, texture_log]])

    # Scale the features using the scaler
    X_new_scaled = scaler.transform(X_new)

    # Make predictions using the loaded ensemble model
    prediction = ensemble_model.predict(X_new_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)

    # Display the result with custom styling
    st.markdown(f'<div class="result">Predicted Result: <span class="result-type"> {predicted_label[0]}</span></div>', unsafe_allow_html=True)
