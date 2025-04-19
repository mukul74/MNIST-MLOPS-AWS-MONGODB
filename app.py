import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Handwritten Digit Recognizer", layout="centered")

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>
        🧠 MNIST Handwritten Digit Recognizer
    </h1>
    <p style='text-align: center; font-size: 18px; color: #666;'>Draw a digit (0–9)
      below and hit Predict!</p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Buttons centered
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.empty()
with col2:
    if st.button("🔍 Predict"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            img = Image.fromarray((img[:, :, 0]).astype("uint8"))
            img = img.resize((28, 28))
            img = np.array(img)
            st.image(img, caption="🖼️ Preprocessed Image", width=140)
            st.success("Model prediction goes here!")  # Hook your model here
        else:
            st.warning("Please draw something before predicting.")
with col3:
    if st.button("🧹 Clear"):
        st.experimental_rerun()
