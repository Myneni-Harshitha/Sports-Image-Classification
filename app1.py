import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from classNames import class_names  # Assuming class_names contains the 100 sports categories

# Load the trained model
model = load_model("./models/Best_Model.h5")

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.7

def resize_image(image, output_size):
    img_resized = image.resize(output_size)
    return img_resized


# Add custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main-title {
            font-size: 2.5em;
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
            margin-bottom: 1em;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 1em;
            padding: 0.5em 1em;
            margin-top: 10px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stFileUploader, .stTextInput {
            font-size: 1.1em;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">Sports Image Classification</div>', unsafe_allow_html=True)

st.header("Upload an Image")

option = st.radio("Choose Image Input Method", ("Upload Image", "Provide URL"))

resized_image = None

if option == "Upload Image":
    image_upload = st.file_uploader("Upload An Image", type=["jpeg", "png"])
    if image_upload is not None:
        img = Image.open(image_upload)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        resized_image = resize_image(img, (224, 224))
elif option == "Provide URL":
    image_url = st.text_input("Enter Image URL")
    btn = st.button("Predict Image")
    if btn:
        if image_url:
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Image from URL", use_column_width=True)
                resized_image = resize_image(img, (224, 224))
            except Exception as e:
                st.error(f"Error downloading image from URL: {e}")
                resized_image = None

if resized_image is not None:
    normalized_image_with_batch = np.expand_dims(resized_image, axis=0)
    detections = model.predict(normalized_image_with_batch)
    
    # Get the predicted class index and confidence
    class_index = np.argmax(detections, axis=1)[0]
    confidence = np.max(detections, axis=1)[0]
    
    # Check if the confidence is above the threshold
    if confidence < CONFIDENCE_THRESHOLD:
        st.error("The model is not confident enough. This image may not belong to a sport category.")
    else:
        # Check if the predicted class is within the expected categories
        if class_index < len(class_names):
            sport_name = class_names[class_index]
            st.success(f"Predicted sport: {sport_name}")
        else:
            st.error("This image does not belong to any known sport category.")
