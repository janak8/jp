# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Define a function to load and compile the model
def load_and_compile_model():
    try:
        model = load_model('CNNModel.h5', compile=False)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'),  # Corrected reduction
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Define class names
CLASS_NAMES = [
    'Tomato-Early_Bright', 'Tomato-Healthy', 'Tomato-Late_bright',
    'Tomato-Leaf_Mold', 'Tomato-Septoria_LeafSpot',
    'Tomato-Spider_Mites', 'Tomato-Target_Spot',
    'Tomato-YellowLeaf-CurlVirus', 'Tomato-Bacterial_spot',
    'Tomato-mosaic_virus'
]

# Add custom CSS for vibrant gradient background, title, and footer
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(45deg, #ff7eb3, #ff758c, #ff6a5e, #ff4500);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    header .stTitle {
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        color: #fff;
    }

    footer {
        visibility: hidden;
    }

    .footer-content {
        visibility: visible;
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>

    <div class="footer-content">
        <p>Developed by Janak Adhikari</p>
        <p>Trusted by Farmers, Agro-Techs, and Colleges</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Set title of the app
st.title("Tomato Leaf Disease Detection")
st.markdown("Upload an image of the plant leaf to detect its disease.")

# Upload the plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict disease when the user uploads an image
if plant_image:
    try:
        # Reload the model every time for fresh predictions
        model = load_and_compile_model()

        # Convert the file to a NumPy array and decode it using OpenCV
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if opencv_image is not None:
            # Display the uploaded image
            st.image(opencv_image, channels="BGR", caption="Uploaded Image")

            # Resize the image to the required dimensions (224x224)
            resized_image = cv2.resize(opencv_image, (224, 224))

            # Expand dimensions to match the model's input shape
            input_image = np.expand_dims(resized_image, axis=0)

            # Normalize the image
            input_image = input_image / 255.0

            # Make prediction
            Y_pred = model.predict(input_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]

            # Display the prediction
            disease, condition = result.split('-')
            st.success(f"This is a {disease} leaf with {condition}.")
        else:
            st.error("Error processing the uploaded image. Please upload a valid image.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload an image to begin disease detection.")
