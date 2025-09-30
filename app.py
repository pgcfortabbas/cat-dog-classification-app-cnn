import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import gdown # <-- DEPENDENCY: Used for downloading the model from Google Drive
import os
from PIL import Image

# --- Configuration Constants ---
# NOTE: This app requires Streamlit, TensorFlow, Pillow, NumPy, and gdown.
# If running in a production environment (like Streamlit Cloud), ensure these
# are listed in your requirements.txt file.
DRIVE_FILE_ID = "1qlCZhwRvsQuJeSmRlHiQXYYqsUdR4h2q" 
MODEL_FILENAME = "saved_model.keras" 
image_size = (128, 128)
CLASS_NAMES = ["Cat", "Dog"] # Assuming 0=Cat, 1=Dog based on typical binary classification setup
CONFIDENCE_THRESHOLD = 0.75

@st.cache_resource
def download_and_load_model():
    """
    Attempts to download the model from Google Drive and load it using Keras.
    Uses st.cache_resource to ensure the heavy download/load process only runs once.
    """
    try:
        # 1. Download the file if it doesn't exist locally
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner(f"Downloading model {MODEL_FILENAME} from Google Drive..."):
                # Download using the provided Google Drive ID
                gdown.download(id=DRIVE_FILE_ID, output=MODEL_FILENAME, quiet=False)
                st.success("Model downloaded successfully!")

        # 2. Load the Keras model
        return tf.keras.models.load_model(MODEL_FILENAME)
    
    except Exception as e:
        # Display detailed error information if loading fails
        st.error(
            f"""
            **MODEL LOAD FAILED!**
            Please check the following:
            1. Is the `DRIVE_FILE_ID` correct in the script?
            2. Is the file shared publicly ("Anyone with the link") on Google Drive?
            3. **CRITICAL:** Ensure the `gdown` dependency is installed in your environment.
            
            Error details: {e}
            """
        )
        st.stop() # Stops the execution of the Streamlit app

st.title("Image Classification App")
st.markdown("---")

# Load the model (or attempt to download and load)
model = download_and_load_model()

st.write("Upload an image and the model will predict its class.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display the image, resizing it to the target size
        # load_img from keras.preprocessing is used here
        image = load_img(uploaded_file, target_size=image_size)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image array
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0 # Add batch dimension and normalize

        # Make prediction
        predictions = model.predict(img_array)
        
        # Determine the predicted class (highest probability index)
        predicted_class_index = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class_index]
        
        st.subheader("Classification Results")

        # Interpret results based on confidence threshold
        if predicted_probability >= CONFIDENCE_THRESHOLD:
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            result_message = f"**{predicted_class_name}**"
            st.success(f"✅ Confident Prediction: {predicted_class_name}")
        else:
            predicted_class_name = "Uncertain/Invalid"
            result_message = "**Uncertain/Invalid** (Confidence too low)"
            st.warning("⚠️ Prediction confidence is low. This image may not be suitable for the model.")

        # Display final results
        st.write(f"Predicted Class: {result_message}")
        st.write(f"Prediction (class index): **{predicted_class_index}**")
        st.write(f"Confidence (Max Probability): **{predicted_probability:.4f}** (Threshold: {CONFIDENCE_THRESHOLD})")

    except Exception as e:
        st.error(f"An error occurred during image processing or prediction: {e}")

# --- Instructions to run the app ---
# 1. Save this code as a Python file (e.g., app.py).
# 2. Before running, ensure all required libraries are installed:
#    pip install streamlit tensorflow pillow numpy gdown
# 3. If deploying, create a requirements.txt file containing these dependencies:
#    streamlit
#    tensorflow
#    pillow
#    numpy
#    gdown
# 4. Open your terminal or command prompt.
# 5. Navigate to the directory where you saved app.py.
# 6. Run the command: streamlit run app.py
