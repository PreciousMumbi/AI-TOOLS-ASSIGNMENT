import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn.h5")

st.title("🖐️ Handwritten Digit Classifier (MNIST)")

st.write("Upload a 28x28 pixel grayscale image of a digit (0-9)")

uploaded_file = st.file_uploader("Choose a PNG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert (if white background)
    image = image.resize((28, 28))
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess image for model
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {predicted_digit}")
    st.write(f"Prediction Confidence: {np.max(prediction) * 100:.2f}%")
