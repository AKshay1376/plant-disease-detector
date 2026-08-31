import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌱",
    layout="centered"
)

# -----------------------------
# Load model and class labels
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

@st.cache_data
def load_labels():
    with open("class_labels.json", "r") as f:
        return json.load(f)

model = load_model()
class_labels = load_labels()

# -----------------------------
# Title
# -----------------------------
st.title("🌱 Plant Disease Detector")
st.write(
    "Upload an image of a plant leaf and the AI will "
    "predict the possible disease."
)

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "📸 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    if st.button("🔍 Detect Disease"):

        with st.spinner("Analyzing the leaf..."):

            # Get model input size automatically
            input_height = model.input_shape[1]
            input_width = model.input_shape[2]

            # Resize image
            image_resized = image.resize(
                (input_width, input_height)
            )

            # Convert image to NumPy array
            image_array = np.array(image_resized)

            # Normalize pixels
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(predictions[0])
            confidence = float(
                np.max(predictions[0]) * 100
            )

            # Get disease name
            predicted_class = class_labels[predicted_index]

        # -----------------------------
        # Display result
        # -----------------------------
        st.success("Prediction Complete! 🌿")

        st.subheader("🦠 Result")

        # Split plant and disease
        if "___" in predicted_class:
            plant, disease = predicted_class.split(
                "___",
                1
            )
        else:
            plant = "Unknown"
            disease = predicted_class

        st.write(f"🌱 **Plant:** {plant}")
        st.write(f"🦠 **Condition:** {disease}")
        st.write(f"📊 **Confidence:** {confidence:.2f}%")

        # Healthy check
        if "healthy" in predicted_class.lower():

            st.info(
                "✅ The leaf appears to be healthy."
            )

        else:

            st.warning(
                "⚠️ A possible plant disease was detected."
            )

            st.write(
                "💡 **Recommendation:** "
                "Consult a local agricultural expert "
                "for treatment appropriate to the crop "
                "and disease."
            )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Plant Disease Detector • AI-powered plant health analysis 🌱"
)
