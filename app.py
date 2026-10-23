import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Medical Image Disease Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Vision Transformer Medical Image Analyzer")
st.markdown("Upload a CT or X-ray image to detect disease status.")

# -----------------------------
# Upload Image FIRST
# -----------------------------
uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg","jpeg","png"])

# -----------------------------
# Load Model (DO NOT STOP APP)
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "medical_vit_final.keras"

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"KerasLayer": hub.KerasLayer},
            compile=False,
            safe_mode=False
        )

        dummy = np.zeros((1, 224, 224, 3))
        model.predict(dummy)

        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# -----------------------------
# If model failed, still show UI
# -----------------------------
if model is None:
    st.warning("⚠ Model not loaded. Upload still visible, but prediction disabled.")

# -----------------------------
# Continue only if image uploaded AND model exists
# -----------------------------
if uploaded_file is not None and model is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption="Uploaded Image", width=350)

    img_array = np.array(image)

    # Preprocess
    img = cv2.resize(img_array,(224,224))
    img = img / 255.0
    img = np.expand_dims(img,axis=0)

    with st.spinner("🔍 Analyzing Image..."):
        prediction = model.predict(img)

    class_names = ['CT_Diseased','CT_Normal','XRAY_Diseased','XRAY_Normal']

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))*100

    predicted_class = class_names[class_index]
    image_type = predicted_class.split("_")[0]
    disease_status = predicted_class.split("_")[1]

    st.subheader("🔍 Prediction Result")

    col1,col2 = st.columns(2)

    with col1:
        st.info(image_type)

    with col2:
        if disease_status == "Diseased":
            st.error("⚠ Diseased")
        else:
            st.success("✅ Normal")

    st.write(f"Confidence: {confidence:.2f}%")