import cv2
import numpy as np

def remove_lighting(image_path, output_path='delighted_output.jpg'):
    """
    Removes lighting and shadows from an input image and saves the de-lighted image.
    
    :param image_path: Path to the input image file.
    :param output_path: Path to save the de-lighted output image. (default: 'delighted_output.jpg')
    :return: The de-lighted image as a numpy array.
    """
    # Load the input image
    img = cv2.imread(image_path)
    
    if img is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    # Convert the image to the LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge the CLAHE enhanced L-channel with the A and B channels
    limg = cv2.merge((cl, a_channel, b_channel))
    
    # Convert the LAB image back to the BGR color space
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Save the output image
    cv2.imwrite(output_path, final_img)
    
    return final_img
