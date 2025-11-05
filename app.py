import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception:
    # Fallback to standalone Keras if TensorFlow is not available in the environment
    from keras.applications import MobileNetV2
    from keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# 🎯 Load model and CNN feature extractor
# -------------------------------
model = joblib.load("best_model_LR.pkl")

# MobileNetV2 feature extractor (same as used during training)
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Define class names (update based on your dataset)
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# -------------------------------
# 🌿 Streamlit UI
# -------------------------------
st.set_page_config(page_title="Plant Disease Classifier 🌿", layout="centered")
st.title("🌿 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to detect its disease using the trained ML model.")

uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to array
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # MobileNetV2 expects 224x224
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract CNN features (length = 1280)
    features = feature_extractor.predict(img)
    st.write("Extracted feature shape:", features.shape)

    # Predict using trained model
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = np.max(prob) * 100
    st.write(f"Prediction confidence: {confidence:.2f}%")

    predicted_class = class_names[int(prediction)] if int(prediction) < len(class_names) else "Unknown"

    st.success(f"✅ Predicted Class: **{predicted_class}**")
