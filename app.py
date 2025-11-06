import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import gdown
import os
from PIL import Image

# --- Configuration Constants ---
DRIVE_FILE_ID = "12Zm-LJUT3_iQW-4Hu1SXnybgHyKXlo04" 
MODEL_FILENAME = "saved_model.keras"
image_size = (128, 128)
# Assign colors to classes for a thematic look
CLASS_INFO = {
    "Cat 🐈": {"index": 0, "color": "#FFC0CB"}, # Pink/Salmon for Cat
    "Dog 🐕": {"index": 1, "color": "#ADD8E6"}  # Light Blue for Dog
}
CLASS_NAMES = list(CLASS_INFO.keys())
CONFIDENCE_THRESHOLD = 0.75

# --- Custom CSS for Theming and Better Layout ---
st.markdown(
    """
    <style>
    /* 1. Main Background */
    .stApp {
        background-color: #f0f2f6; /* Very light gray */
    }
    /* 2. Custom Metric Styling */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* 3. Center the Title and Image */
    .stTitle, .stImage {
        text-align: center;
    }
    /* 4. Custom Progress Bar Background (Simulating Confidence) */
    .confidence-bar-bg {
        background-color: #e0e0e0;
        border-radius: 5px;
        height: 25px;
        overflow: hidden;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .confidence-bar-fill {
        height: 100%;
        text-align: center;
        line-height: 25px;
        color: white;
        font-weight: bold;
        transition: width 0.5s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Utility Functions ---

@st.cache_resource
def download_and_load_model():
    """Attempts to download and load the model."""
    try:
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner(f"Downloading model {MODEL_FILENAME} from Google Drive..."):
                gdown.download(id=DRIVE_FILE_ID, output=MODEL_FILENAME, quiet=False)
                st.success("Model downloaded successfully!")
        return tf.keras.models.load_model(MODEL_FILENAME)
    except Exception as e:
        st.error(f"**MODEL LOAD FAILED!** Error: {e}")
        st.stop()

def display_confidence_bar(probability, predicted_class_name):
    """Generates the custom HTML for a themed confidence bar."""
    
    # Determine the color based on the predicted class
    class_key = next((k for k in CLASS_INFO if k == predicted_class_name), None)
    fill_color = CLASS_INFO[class_key]['color'] if class_key else '#808080'
    
    # Format the width percentage
    width_percent = f"{probability * 100:.2f}%"

    html = f"""
    <div class="confidence-bar-bg">
        <div class="confidence-bar-fill" style="width: {width_percent}; background-color: {fill_color};">
            Confidence: {width_percent}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- Streamlit UI Layout and Logic ---

# 1. Set Page Configuration
st.set_page_config(
    page_title="Pet Classifier 🐶 vs 🐱",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapse the sidebar initially for a clean start
)

## 🌟 Thematic Title
st.title("🐾 Thematic Cat vs. Dog Classifier")
st.markdown("A deep learning demo with visual confidence metering.")

# Load the model
with st.spinner("Initializing the Keras model..."):
    model = download_and_load_model()
st.sidebar.success("Model is ready!")

st.markdown("---")

col_upload, col_result = st.columns([1, 2]) # Wider column for results

# --- Left Column: Upload and Model Info ---
with col_upload:
    st.subheader("1. Upload & View")
    uploaded_file = st.file_uploader("Choose a Cat or Dog image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", width=300) 
        
        # Prediction button only appears AFTER an image is uploaded
        predict_button = st.button("🚀 Classify Image", use_container_width=True, type="primary")
    else:
        st.info("⬆️ Upload an image to start classification.")
        predict_button = False

    st.markdown("---")
    st.caption("Model Parameters")
    st.markdown(f"* Input Size: `{image_size[0]}x{image_size[1]}`")
    st.markdown(f"* Confidence Threshold: `{CONFIDENCE_THRESHOLD * 100:.0f}%`")


# --- Right Column: Results ---
with col_result:
    st.subheader("2. Prediction Analysis")
    
    if uploaded_file and predict_button:
        with st.spinner("Analyzing image..."):
            try:
                # Preprocess the image
                image = load_img(uploaded_file, target_size=image_size)
                img_array = img_to_array(image)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Make prediction
                predictions = model.predict(img_array, verbose=0)
                predicted_index = np.argmax(predictions)
                predicted_probability = predictions[0][predicted_index]
                predicted_class_name = CLASS_NAMES[predicted_index]
                
                # --- Final Result Highlight ---
                if predicted_probability >= CONFIDENCE_THRESHOLD:
                    st.balloons() # Celebration effect for a confident result
                    st.success(f"✅ HIGH CONFIDENCE: It looks like a **{predicted_class_name}**!")
                else:
                    st.warning("⚠️ LOW CONFIDENCE: The model is uncertain about the class.")
                
                # Use a custom HTML bar for visualization
                display_confidence_bar(predicted_probability, predicted_class_name)

                st.metric(
                    label="🥇 Highest Probability Class", 
                    value=predicted_class_name, 
                    delta=f"Confidence: {predicted_probability*100:.2f}%"
                )

                st.markdown("### Detailed Scores")
                # Display detailed scores using columns for a clear side-by-side comparison
                
                col_cat, col_dog = st.columns(2)
                
                # CAT SCORE
                with col_cat:
                    cat_prob = predictions[0][CLASS_INFO["Cat 🐈"]['index']]
                    cat_color = CLASS_INFO["Cat 🐈"]['color']
                    
                    st.markdown(f"""
                        <div style='background-color: {cat_color}; padding: 10px; border-radius: 5px;'>
                            <h4 style='color: #333;'>Cat 🐈 Score</h4>
                            <p style='font-size: 24px; font-weight: bold;'>{cat_prob * 100:.2f}%</p>
                            <progress value='{cat_prob}' max='1.0' style='width: 100%;'></progress>
                        </div>
                    """, unsafe_allow_html=True)
                
                # DOG SCORE
                with col_dog:
                    dog_prob = predictions[0][CLASS_INFO["Dog 🐕"]['index']]
                    dog_color = CLASS_INFO["Dog 🐕"]['color']
                    
                    st.markdown(f"""
                        <div style='background-color: {dog_color}; padding: 10px; border-radius: 5px;'>
                            <h4 style='color: #333;'>Dog 🐕 Score</h4>
                            <p style='font-size: 24px; font-weight: bold;'>{dog_prob * 100:.2f}%</p>
                            <progress value='{dog_prob}' max='1.0' style='width: 100%;'></progress>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.info(f"Threshold Check: Probability {predicted_probability:.4f} vs. {CONFIDENCE_THRESHOLD}")


            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    elif uploaded_file:
        st.info("Click the '🚀 Classify Image' button to see the results.")
    else:
        st.markdown(
            """
            <div style='text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 10px;'>
                <p style='font-size: 20px; color: #888;'>Awaiting image upload...</p>
                <p style='color: #888;'>Once uploaded, click 'Classify Image' for analysis.</p>
            </div>
            """, unsafe_allow_html=True
        )

# --- End of App ---
