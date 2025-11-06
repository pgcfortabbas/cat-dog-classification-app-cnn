import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import gdown
import os
from PIL import Image

# --- Configuration Constants ---
DRIVE_FILE_ID = "12Zm-LJUT3_iQW-4Hu1SXnybgHyKXlo04" # UPDATED FILE ID
MODEL_FILENAME = "saved_model.keras"
image_size = (128, 128)
CLASS_NAMES = ["Cat 🐈", "Dog 🐕"] # Added emojis for better visual flair
CONFIDENCE_THRESHOLD = 0.75

# --- Utility Functions ---
@st.cache_resource
def download_and_load_model():
    """
    Attempts to download the model from Google Drive and load it using Keras.
    """
    try:
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner(f"Downloading model {MODEL_FILENAME} from Google Drive..."):
                gdown.download(id=DRIVE_FILE_ID, output=MODEL_FILENAME, quiet=False)
                st.success("Model downloaded successfully!")

        return tf.keras.models.load_model(MODEL_FILENAME)

    except Exception as e:
        st.error(
            f"""
            **MODEL LOAD FAILED!**
            **Critical:** Ensure the `gdown` dependency is installed (`pip install gdown`).
            Error details: {e}
            """
        )
        st.stop()

# --- Streamlit UI Layout and Logic ---

# 1. Set Page Configuration for a wider layout
st.set_page_config(
    page_title="Pet Classifier 🐶 vs 🐱",
    layout="wide",
    initial_sidebar_state="auto"
)

## 🌟 Pet Classifier App 🌟
st.title("🐾 Cat vs. Dog Classifier (CNN Demo)")
st.markdown("Upload an image to instantly classify it as a Cat or a Dog using a pre-trained Keras model.")

# Load the model
with st.spinner("Initializing the model..."):
    model = download_and_load_model()
st.sidebar.success("Model is ready!")

# --- Main Content Area ---
st.markdown("---")

col1, col2 = st.columns([1, 2]) # Use columns for a better side-by-side layout

with col1:
    ## 🖼️ Upload Your Image
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    # Simple model info in the sidebar
    st.sidebar.subheader("Model Info")
    st.sidebar.write(f"Input Size: **{image_size[0]}x{image_size[1]}**")
    st.sidebar.write(f"Classes: **{', '.join([name.split()[0] for name in CLASS_NAMES])}**")
    st.sidebar.write(f"Confidence Threshold: **{CONFIDENCE_THRESHOLD * 100:.0f}%**")


with col2:
    st.subheader("2. Classification Results")
    if uploaded_file is not None:
        try:
            # Load and display the image (keeping the original ratio)
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Image", width=300) # Fixed width for cleaner look

            # Preprocess the image
            image = load_img(uploaded_file, target_size=image_size)
            img_array = img_to_array(image)
            # Add batch dimension and normalize
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make prediction
            predictions = model.predict(img_array, verbose=0) # verbose=0 to silence output
            predicted_class_index = np.argmax(predictions)
            predicted_probability = predictions[0][predicted_class_index]
            
            # Use Streamlit's expander for a cleaner look of the "Predict" action
            with st.expander("▶️ Run Prediction"):
                st.write(f"Raw Prediction Array: `{predictions.tolist()}`")
                
            # --- Results Display ---
            if predicted_probability >= CONFIDENCE_THRESHOLD:
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                
                # Use a metric to highlight the main result
                st.metric(label="🏆 Final Classification", 
                          value=predicted_class_name, 
                          delta=f"{predicted_probability*100:.2f}% Confidence")
                
                st.success(f"**Confident Prediction:** The model is **{predicted_class_name.split()[0]}**!")
            else:
                predicted_class_name = "Uncertain/Low Confidence"
                st.metric(label="❌ Final Classification",
                          value=predicted_class_name,
                          delta=f"{predicted_probability*100:.2f}% Confidence")
                
                st.warning("⚠️ **Prediction confidence is low.** This image might be too complex, corrupted, or not a clear Cat/Dog.")

            # Display all probabilities in a table/chart for completeness
            st.markdown("### Class Probabilities")
            # Create a simple DataFrame or list for the chart/table
            
            prob_data = {
                "Class": [name.split()[0] for name in CLASS_NAMES],
                "Probability": [p * 100 for p in predictions[0]]
            }
            
            # Display a bar chart
            st.bar_chart(prob_data, x='Class', y='Probability', color="#ff4b4b") # Use a Streamlit-friendly color

        except Exception as e:
            st.error(f"An error occurred during image processing or prediction: {e}")

    else:
        st.info("Please upload an image file (JPG, JPEG, or PNG) to get started.")
        st.image("https://images.unsplash.com/photo-1517423738875-5ceee1aa200f?q=80&w=200&h=200&fit=crop", 
                 caption="Waiting for an image...", width=300)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
    }
    </style>
    """, unsafe_allow_html=True
)
