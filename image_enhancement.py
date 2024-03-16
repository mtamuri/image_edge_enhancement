import cv2
import streamlit as st
import numpy as np

def edge_enhancement(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply the Sobel operator
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the result
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Display the original and enhanced images
    cv2.imshow("Original Image", image)
    cv2.imshow("Enhanced Image", sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

st.title('Welcome to image edge enhancement!\nWebApp built for image processing by: [Ammar MT.](https://ammarmt.tech/) ©', anchor='center')


st.write(
    'Please select the technique you want to use to enhance the image. \n'
)

edge_enhancements = {
    'Sobel Operator': 'Sobel Operator',
    'Laplacian Operator': 'Laplacian Operator',
    'Canny Edge Detection': 'Canny Edge Detection',
    'Histogram Equalization': 'Histogram Equalization',
    'Contrast Stretching': 'Contrast Stretching'
}

selected_edge_enhancement = st.selectbox( "select a technique of image enhancment:", edge_enhancements.keys())

if selected_edge_enhancement:
    st.write(f"You selected: {selected_edge_enhancement}")


# Path to the image
#image_path = "upwork.jpg"

# Call the edge_enhancement function
#edge_enhancement(image_path)
