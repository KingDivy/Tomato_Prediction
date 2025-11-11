import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random

try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception:
    from keras.applications import MobileNetV2
    from keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# 🎯 Load model and CNN feature extractor
# -------------------------------
model = joblib.load("best_model_LR.pkl")
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# -------------------------------
# 🌱 Class Names
# -------------------------------
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
# 🩺 Disease Info Dictionary
# -------------------------------
disease_info = {
    "Tomato___Bacterial_spot": {
        "Description": "Dark, water-soaked lesions on leaves and fruits caused by Xanthomonas bacteria.",
        "Solution": "Use copper-based bactericides. Remove infected plants and avoid overhead watering."
    },
    "Tomato___Early_blight": {
        "Description": "Fungal disease causing concentric rings on older leaves (Alternaria solani).",
        "Solution": "Apply fungicides with chlorothalonil or copper. Remove affected leaves and rotate crops."
    },
    "Tomato___Late_blight": {
        "Description": "Spreads rapidly under humidity, causing brown patches with gray mold.",
        "Solution": "Use fungicides with mancozeb or metalaxyl. Remove infected plants immediately."
    },
    "Tomato___Leaf_Mold": {
        "Description": "Yellow spots on upper surfaces and olive-green mold underneath.",
        "Solution": "Ensure air circulation, avoid overcrowding, and use fungicides like chlorothalonil."
    },
    "Tomato___Septoria_leaf_spot": {
        "Description": "Small circular spots with dark borders, often on lower leaves.",
        "Solution": "Prune infected leaves and apply fungicides. Avoid overhead watering."
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "Description": "Mites suck sap from leaves, leaving yellow specks and webbing.",
        "Solution": "Spray neem oil or insecticidal soap. Maintain humidity to deter mites."
    },
    "Tomato___Target_Spot": {
        "Description": "Circular spots with concentric rings on leaves and fruits.",
        "Solution": "Use preventive fungicides and keep the area clean of debris."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "Description": "Viral disease causing leaf curling and yellowing, spread by whiteflies.",
        "Solution": "Control whiteflies using insecticides and plant resistant varieties."
    },
    "Tomato___Tomato_mosaic_virus": {
        "Description": "Leads to mottling, curling, and stunted plant growth.",
        "Solution": "Use virus-free seeds and disinfect tools regularly."
    },
    "Tomato___healthy": {
        "Description": "Leaf appears healthy with no visible infection or deformity.",
        "Solution": "Maintain proper watering, fertilization, and pest monitoring."
    }
}

# -------------------------------
# 🌿 Streamlit UI Configuration
# -------------------------------
st.set_page_config(page_title="Tomato Leaf Disease Classifier 🌿", layout="centered")

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 10px;
        padding: 15px;
    }
    h1 {
        color: #145A32;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #16a085;
        color: white;
        border-radius: 8px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍅 Tomato Leaf Disease Detection")
st.caption("Upload a tomato leaf image to detect its disease using the trained Logistic Regression model (91% accuracy).")

uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract features
    features = feature_extractor.predict(img)

    # Predict
    prob = model.predict_proba(features)[0]
    prediction = np.argmax(prob)
    confidence = np.max(prob) * 100
    predicted_class = class_names[prediction]

    # Result Display
    st.success(f"✅ Predicted Class: **{predicted_class}**")
    st.progress(int(confidence))
    st.info(f"🔹 Model Confidence: {confidence:.2f}%")

    # Disease Info
    if predicted_class in disease_info:
        info = disease_info[predicted_class]
        st.markdown("### 🩺 Disease Information")
        st.write(f"**Description:** {info['Description']}")
        st.write(f"**Suggested Solution:** {info['Solution']}")

    # Confidence Breakdown Chart
    st.markdown("### 📊 Confidence by Disease Class")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(class_names, prob, color="green")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Disease Class")
    plt.tight_layout()
    st.pyplot(fig)

    # Download Report
    report = f"""
    🍅 Tomato Leaf Disease Detection Report
    ---------------------------------------
    Predicted Class: {predicted_class}
    Confidence: {confidence:.2f}%
    Description: {info['Description']}
    Suggested Solution: {info['Solution']}
    """
    st.download_button("📥 Download Diagnosis Report", report)

    # Random Farming Tip
    tips = [
        "Rotate tomato crops every 2 years to prevent soil-borne diseases.",
        "Avoid overhead watering to reduce fungal infections.",
        "Use neem oil as a natural pesticide for mite control.",
        "Check leaves weekly to detect diseases early.",
        "Use disease-resistant tomato varieties whenever possible."
    ]
    st.markdown("### 🌾 Pro Farming Tip")
    st.success(random.choice(tips))
