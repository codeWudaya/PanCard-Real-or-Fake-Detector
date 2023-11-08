import streamlit as st
from PIL import Image
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to calculate SSIM
def calculate_ssim(image1, image2):
    return ssim(image1, image2)

# Function to compare images and display result
def compare_images(real_image, user_image):
    real_gray = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    user_gray = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)
    
    ssim_value = calculate_ssim(real_gray, user_gray)
    
    return ssim_value


# Streamlit app
st.title("PAN Card Real vs Fake Detector")

# Upload user's PAN card image
user_image = st.file_uploader("Upload your PAN Card Image", type=["jpg", "png"])

if user_image is not None:
    # Load known real PAN card image
    real_image_path = 'images after resizing and format changing/real.png'
    real_image = cv2.imread(real_image_path)
    
    # Resize user uploaded image to match the dimensions of the real image
    uploaded_image = Image.open(user_image)
    user_image_resized = uploaded_image.resize((real_image.shape[1], real_image.shape[0]))
    
    # Display user's uploaded image
    st.image(user_image_resized, caption='Uploaded PAN Card Image', use_column_width=True)
    
    # Convert user uploaded image to OpenCV format
    user_image_cv = cv2.cvtColor(np.array(user_image_resized), cv2.COLOR_RGB2BGR)
    
    # Compare images
    ssim_value = compare_images(real_image, user_image_cv)
    
    # Display SSIM value
    st.write(f"SSIM Value: {ssim_value}")
    
    # Determine if the PAN card is real or fake based on SSIM value (you may need to adjust this threshold)
    threshold = 0.9
    if ssim_value >= threshold:
        st.write("Result: Genuine PAN Card")
        st.balloons()
    else:
        st.write("Result: Potentially Fake PAN Card")