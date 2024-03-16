import cv2
import streamlit as st
import numpy as np

def sobel_operator(image):
    # Apply the Sobel operator
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the result
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return sobel

def laplacian_operator(image):
    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Normalize the result
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return laplacian

def canny_edge_detection(image):
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)

    return edges

def histogram_equalization(image):
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)

    return equalized_image

image_enhancement_explanations = {
    'Sobel Operator': 'The Sobel operator is a popular edge detection algorithm used to find edges in an image. It computes the gradient magnitude at each pixel by convolving the image with small kernels in both horizontal and vertical directions.',
    'Laplacian Operator': 'The Laplacian operator is an edge detection algorithm that computes the second derivative of the image to identify regions of rapid intensity change, which indicate edges. It highlights areas of high curvature or sharp intensity transitions.',
    'Canny Edge Detection': 'Canny edge detection is a multi-stage algorithm used to detect a wide range of edges in an image while reducing noise and preserving edge details. It involves gradient calculation, non-maximum suppression, and edge tracking by hysteresis.',
    'Histogram Equalization': 'Histogram equalization is a technique used to enhance the contrast of an image by redistributing pixel intensities to cover the entire dynamic range. It computes the cumulative distribution function (CDF) of pixel intensities and maps them to new values for improved contrast.'
}

st.title('Welcome to image edge enhancement!\nWebApp built for image processing by: [Ammar MT.](https://ammarmt.tech/) ©', anchor='center')

st.write('Please select the technique you want to use to enhance the image.')

edge_enhancements = {
    'Sobel Operator': sobel_operator,
    'Laplacian Operator': laplacian_operator,
    'Canny Edge Detection': canny_edge_detection,
    'Histogram Equalization': histogram_equalization,
}

selected_edge_enhancement = st.selectbox("Select a technique of image enhancement:", list(image_enhancement_explanations.keys()))

if selected_edge_enhancement:
    st.write(f'You selected {selected_edge_enhancement}')
    st.write(f"Explanation: {image_enhancement_explanations[selected_edge_enhancement]}")
    
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        enhanced_image = edge_enhancements[selected_edge_enhancement](original_image)

        # Display original and enhanced images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.header("Original Image")
            st.image(original_image, caption='Original Image', use_column_width=True)

        with col2:
            st.header("Enhanced Image")
            st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)
