import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Path to the model
model_path = 'models/mymodel.h5'

# Load the model
def load_and_compile_model():
    try:
        model = load_model(model_path)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Function to predict disease based on model output
def predict_disease(model, image_path):
    # Load and preprocess the image
    test_image = load_img(image_path, target_size=(128, 128))  # Resize image to (128, 128) based on model's expected input
    test_image = img_to_array(test_image) / 255.0  # Normalize image as during training
    test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions for batch input

    # Make prediction
    result = model.predict(test_image)
    pred = np.argmax(result, axis=1)  # Get the predicted class

    # Conditional statements based on prediction
    if pred == 0:
        return "Tomato - Bacterial Spot Disease", 'Tomato-Bacterial_Spot.html'
    elif pred == 1:
        return "Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'
    elif pred == 2:
        return "Tomato - Healthy and Fresh", 'Tomato-Healthy.html'
    elif pred == 3:
        return "Tomato - Late Blight Disease", 'Tomato-Late_Blight.html'
    elif pred == 4:
        return "Tomato - Leaf Mold Disease", 'Tomato-Leaf_Mold.html'
    elif pred == 5:
        return "Tomato - Septoria Leaf Spot Disease", 'Tomato-Septoria_Leaf_Spot.html'
    elif pred == 6:
        return "Tomato - Target Spot Disease", 'Tomato-Target_Spot.html'
    elif pred == 7:
        return "Tomato - Yellow Leaf Curl Virus Disease", 'Tomato-Yellow_Leaf_Curl_Virus.html'
    elif pred == 8:
        return "Tomato - Mosaic Virus Disease", 'Tomato-Mosaic_Virus.html'
    elif pred == 9:
        return "Tomato - Spider Mite Disease", 'Tomato-Spider_Mite.html'

# Add custom CSS for leaf design and animation
st.markdown(
    """
    <style>
    /* Background style */
    .stApp {
        background-color: #e0f7fa;
        overflow: hidden;
        position: relative;
    }

    /* Leaf animation keyframe */
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    /* Styling the leaf shape */
    .leaf {
        width: 100px;
        height: 200px;
        background: linear-gradient(135deg, #4caf50, #81c784);
        position: absolute;
        top: 50%;
        left: 50%;
        transform-origin: center;
        border-radius: 50% 50% 0 0;
        animation: float 4s ease-in-out infinite;
        box-shadow: 4px 4px 20px rgba(0, 0, 0, 0.1);
        z-index: 10;
    }

    /* Adding veins for realism */
    .leaf:before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 4px;
        height: 60%;
        background-color: #388e3c;
        transform: translateX(-50%);
    }

    /* Adding another vein for realism */
    .leaf:after {
        content: '';
        position: absolute;
        top: 40%;
        left: 50%;
        width: 4px;
        height: 50%;
        background-color: #388e3c;
        transform: translateX(-50%) rotate(45deg);
    }

    /* Title and footer styles */
    header .stTitle {
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        color: #333;
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
    """, unsafe_allow_html=True
)

# Display the floating leaf in the center of the screen
st.markdown(
    """
    <div class="leaf"></div>
    """, unsafe_allow_html=True
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

            # Save the image to a temporary file to use for prediction
            temp_image_path = "temp_image.jpg"
            cv2.imwrite(temp_image_path, opencv_image)

            # Make prediction using the model
            pred, output_page = predict_disease(model, temp_image_path)

            # Display the prediction
            st.success(f"This is a {pred}.")
            st.markdown(f"Learn more about this disease: [Click here](static/{output_page})")

        else:
            st.error("Error processing the uploaded image. Please upload a valid image.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload an image to begin disease detection.")
