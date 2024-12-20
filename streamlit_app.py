import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the saved CNN model
MODEL_PATH = 'our_model.keras'
model = load_model(MODEL_PATH)

# Constants
CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Page Layout
st.set_page_config(page_title="Image Classifier", layout="wide", page_icon="🚀")

# Title and Header
st.title("🚀 **Image Classification Web App**")
st.write(
    """
    Upload an image, and this web app will classify it into one of the following categories:
    """
)
st.write(", ".join(CLASS_NAMES))

# Sidebar for File Upload
st.sidebar.title("Upload Image")
st.sidebar.write("Please upload an image in **.jpg** or **.png** format.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predictions
    with st.spinner("🤖 Analyzing the image..."):
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

    # Display results
    st.success(f"🎉 **Predicted Class:** {CLASS_NAMES[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    # Visualize prediction probabilities
    st.subheader("🔍 Class Probabilities")
    probabilities = dict(zip(CLASS_NAMES, predictions[0]))
    st.bar_chart(probabilities)

# Footer
st.markdown("---")
st.markdown(
    """
    🧑‍💻 Developed with ❤️ by Group 28   
    Hope You loved it!!
    """
)