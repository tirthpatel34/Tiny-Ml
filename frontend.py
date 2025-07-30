# frontend.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        os.path.join(os.getcwd(), "tomato_health_best_l2.h5")
    )

model = load_model()
IMG_SIZE = (128, 128)

st.set_page_config(page_title="Tomato Leaf Detector")
st.title("🍅 Tomato Leaf Bacterial Spot Detector")

uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your input image", use_column_width=True)

    st.write("Classifying…")
    x = img.resize(IMG_SIZE)
    x = np.array(x) / 255.0
    x = np.expand_dims(x, 0).astype(np.float32)

    pred = model.predict(x)[0][0]
    label = "Bacterial Spot 🍂" if pred > 0.5 else "Healthy ✅"
    conf  = pred if pred > 0.5 else 1 - pred

    st.markdown(f"## {label}")
    st.write(f"Confidence: **{conf*100:.1f}%**")
