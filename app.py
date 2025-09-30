import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Instructions on how to run the app:
# 1. Save this code as a Python file (e.g., app.py).
# 2. Make sure you have Streamlit and TensorFlow installed: pip install streamlit tensorflow pillow numpy
# 3. Make sure the model file 'cat_dog_classifier_model.keras' is in the same directory as this app.py file, or provide the correct path.
# 4. Open your terminal or command prompt.
# 5. Navigate to the directory where you saved app.py.
# 6. Run the command: streamlit run app.py

# Load the trained model
# Make sure the path below is correct relative to where you run the streamlit app,
# or use the absolute path.
model_path = 'https://drive.google.com/file/d/12Zm-LJUT3_iQW-4Hu1SXnybgHyKXlo04/view?usp=sharing'

@st.cache_resource
def load_my_model(model_path):
    """Loads the trained Keras model."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_my_model(model_path)

st.title("Dog or Cat Image Classifier")
st.write("Upload an image and the model will predict if it's a dog or a cat.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image for the model
        img_height, img_width = 128, 128 # Ensure this matches the training size
        img_array = image.resize((img_width, img_height))
        img_array = tf.keras.preprocessing.image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Rescale pixel values

        # Make prediction
        prediction = model.predict(img_array)
        score = prediction[0][0]

        # Interpret the prediction
        if score > 0.5:
            st.write(f"Prediction: Dog ({score:.2f})")
        elif score <= 0.5:
            st.write(f"Prediction: Cat ({1-score:.2f})")
        else:
             st.write("Prediction: Invalid Image") # This case is unlikely with a sigmoid output, but included for completeness

    except Exception as e:
        st.error(f"Error processing image or making prediction: {e}")
