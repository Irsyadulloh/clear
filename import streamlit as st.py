import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to adjust brightness
def adjust_brightness(image, brightness=0):
    return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

# Function for rotation
def rotate_image(image, angle=0):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

# Function for scaling
def scale_image(image, scale_factor=1.0):
    rows, cols = image.shape[:2]
    return cv2.resize(image, (int(cols * scale_factor), int(rows * scale_factor)))

# Function for translation
def translate_image(image, tx=0, ty=0):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (cols, rows))

# Function for skewing
def skew_image(image, sx=0, sy=0):
    rows, cols = image.shape[:2]
    M = np.float32([[1, sx, 0], [sy, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows))

# Streamlit UI
st.title("Image Processing App")

# Upload Image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpeg"])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Brightness slider
    brightness = st.slider("Adjust Brightness", -100, 100, 0)

    # Transformation sliders
    rotation_angle = st.slider("Rotate Image (degrees)", -180, 180, 0)
    scale_factor = st.slider("Scale Image", 0.1, 3.0, 1.0)
    translation_x = st.slider("Translate X (pixels)", -200, 200, 0)
    translation_y = st.slider("Translate Y (pixels)", -200, 200, 0)
    skew_x = st.slider("Skew X", -0.5, 0.5, 0.0)
    skew_y = st.slider("Skew Y", -0.5, 0.5, 0.0)

    # Apply transformations
    img_bright = adjust_brightness(image, brightness)
    img_rotated = rotate_image(img_bright, rotation_angle)
    img_scaled = scale_image(img_rotated, scale_factor)
    img_translated = translate_image(img_scaled, translation_x, translation_y)
    img_skewed = skew_image(img_translated, skew_x, skew_y)

    # Display transformed image
    st.image(img_skewed, caption="Transformed Image", use_column_width=True)

    # Display before and after images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")
    ax2.imshow(img_skewed)
    ax2.set_title("Processed Image")
    ax2.axis("off")
    st.pyplot(fig)
