import io

import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Set responsive layout
st.set_page_config(page_title="MNIST Handwritten Digit Recognizer", layout="wide")

# Title - responsive HTML styling
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

# Centered layout for canvas using columns
canvas_col1, canvas_col2, canvas_col3 = st.columns([1, 2, 1])
with canvas_col2:
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

st.markdown("")

# Predict and Clear buttons
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
with btn_col2:
    predict = st.button("🔍 Predict", use_container_width=True)
    clear = st.button("🧹 Clear", use_container_width=True)

if predict:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = Image.fromarray(img.astype("uint8")).convert("L")
        img = Image.eval(img, lambda x: 255 - x)
        img = img.resize((28, 28))

        st.image(img, caption="🖼️ Preprocessed Image", width=140)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        try:
            response = requests.post(
                "http://localhost:8000/predict_image",
                files={"file": ("image.png", img_bytes, "image/png")},
            )
            if response.ok:
                prediction = response.json().get("predicted_digit", "Unknown")
                st.success(f"🧠 Model Prediction: **{prediction}**")
            else:
                st.error("❌ Failed to get prediction from backend.")
        except Exception as e:
            st.error(f"🚫 Error connecting to API: {e}")
    else:
        st.warning("Please draw something before predicting.")

if clear:
    st.rerun()
