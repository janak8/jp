# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Load the model
try:
    model = load_model('CNNModel.h5', compile=False)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'),  # Corrected reduction
        metrics=['accuracy']
    )
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

# Set title of the app
st.title("Tomato Leaf Disease Detection")
st.markdown("Upload an image of the plant leaf to detect its disease.")

# Upload the plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict disease when the user uploads an image
if plant_image:
    try:
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
