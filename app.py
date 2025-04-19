import io

import requests
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
            # Convert canvas to image (grayscale)
            img = canvas_result.image_data  # Should be (H, W, 4) RGBA
            # Convert RGBA to grayscale
            img = Image.fromarray(img.astype("uint8")).convert("L")
            # Invert (if needed)
            img = Image.eval(img, lambda x: 255 - x)
            # Resize to match model input
            img = img.resize((28, 28))
            # Display the image
            st.image(img, caption="🖼️ Preprocessed Image", width=140)

            # Convert to bytes
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            try:
                # Send to FastAPI endpoint
                response = requests.post(
                    "http://localhost:8001/predict_image",  # Change if your backend is hosted elsewhere
                    files={"file": ("image.png", img_bytes, "image/png")},
                )
                print("Response in app.py :", response.json())
                if response.ok:
                    prediction = response.json().get("predicted_digit", "Unknown")
                    print("Prediction in app.py :", prediction)
                    st.success(f"🧠 Model Prediction: **{prediction}**")
                else:
                    st.error("❌ Failed to get prediction from backend.")

            except Exception as e:
                st.error(f"🚫 Error connecting to API: {e}")

        else:
            st.warning("Please draw something before predicting.")

with col3:
    if st.button("🧹 Clear"):
        st.experimental_rerun()
