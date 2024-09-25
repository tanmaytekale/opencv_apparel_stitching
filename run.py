"""import cv2
import numpy as np

# Correct paths to your images and reference image
image_paths = [
    'D:/polycosmos/projects/opencv/data/gray_dress/1.jpg',
    'D:/polycosmos/projects/opencv/data/gray_dress/2.jpg',
    'D:/polycosmos/projects/opencv/data/gray_dress/3.jpg',
    'D:/polycosmos/projects/opencv/data/gray_dress/4.jpg',
    'D:/polycosmos/projects/opencv/data/gray_dress/5.jpg',
    'D:/polycosmos/projects/opencv/data/gray_dress/6.jpg',
    'D:/polycosmos/projects/opencv/data/gray_dress/7.jpg',
    'D:/polycosmos/projects/opencv/data/gray_dress/8.jpg'
]

reference_image_path = 'D:/polycosmos/projects/opencv/data/gray_dress/reference.jpg'

# Load the reference image
reference_image = cv2.imread(reference_image_path)

# Check if reference image is loaded
if reference_image is None:
    print("Error: Reference image couldn't be loaded.")
    exit()

# Resize images (optional)
def resize_image(img, scale_percent=40):
    if img is None:
        return None
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# Resize reference image to a smaller size for faster processing (optional)
reference_image = resize_image(reference_image)

# Feature detection and alignment function using reference image
def align_images_using_reference(img, ref_img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_img, des_img = sift.detectAndCompute(gray_img, None)
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)

    # FLANN-based matcher for feature matching
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_img, des_ref, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Homography matrix to align the images
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        height, width = ref_img.shape[:2]
        aligned_img = cv2.warpPerspective(img, M, (width, height))

        return aligned_img, M
    else:
        print("Not enough matches found.")
        return None, None

# Stitch the aligned images together using maximum blending
def stitch_images(base_img, img_to_add):
    return np.maximum(base_img, img_to_add)

# Load and process each image
stitched_result = None

for image_path in image_paths:
    image = cv2.imread(image_path)

    # Check if image loaded properly
    if image is None:
        print(f"Error: {image_path} couldn't be loaded.")
        exit()

    # Optionally resize image for faster processing
    image = resize_image(image)

    # Align image to the reference
    aligned_image, _ = align_images_using_reference(image, reference_image)

    if aligned_image is None:
        print(f"Error: Alignment failed for {image_path}")
        exit()

    # Initialize or stitch images together
    if stitched_result is None:
        stitched_result = aligned_image
    else:
        stitched_result = stitch_images(stitched_result, aligned_image)

# Save the final stitched image
output_path = 'D:/polycosmos/projects/opencv/data/gray_dress/stitched_output.jpg'
cv2.imwrite(output_path, stitched_result)

print(f"Stitched image saved as {output_path}")
"""























import cv2
import os
import time
import numpy as np
from delight import remove_lighting

# Set this variable to True to enable de-lighting, False to disable
ENABLE_DELIGHT = False  # <<< Edit this line to toggle delighting

# Directories for inputs, reference, delights, and outputs
input_dir = 'D:\polycosmos\projects\opencv\data/raw'
reference_dir = 'D:\polycosmos\projects\opencv\data/reference'
delight_dir = 'D:\polycosmos\projects\opencv\data/delight'
output_dir = 'D:\polycosmos\projects\opencv\data/output'

# Helper function to delight images if enabled
def process_image(image_path):
    if ENABLE_DELIGHT:
        # Get image filename and construct delight output path
        image_filename = os.path.basename(image_path)
        delight_output_path = os.path.join(delight_dir, image_filename)
        
        # Time the de-lighting process
        start_time = time.time()
        
        # Run delight function and return the processed image path
        print(f"Processing image for delight: {image_filename}")
        delighted_image = remove_lighting(image_path, delight_output_path)
        
        end_time = time.time()
        print(f"De-lighting completed for {image_filename} in {end_time - start_time:.2f} seconds.")
        
        return delight_output_path
    else:
        # If delight is disabled, return the original image path
        return image_path

# Load and process all input images from the 'raw' folder
input_images = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if img.endswith('.jpg')]
processed_images = [process_image(img) for img in input_images]

# Load the reference image
reference_image_path = os.path.join(reference_dir, 'reference.jpg')
reference_image = cv2.imread(reference_image_path)

# Resize images (optional)
def resize_image(img, scale_percent=40):
    if img is None:
        return None
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# Resize reference image
reference_image = resize_image(reference_image)

# Align the images using the reference image
def align_images_using_reference(img, ref_img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_img, des_img = sift.detectAndCompute(gray_img, None)
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_img, des_ref, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        height, width = ref_img.shape[:2]
        aligned_img = cv2.warpPerspective(img, M, (width, height))

        return aligned_img, M
    else:
        print("Not enough matches found.")
        return None, None

# Load and align all images
aligned_images = []
for img_path in processed_images:
    print(f"Loading image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        continue
    img = resize_image(img)
    aligned_img, _ = align_images_using_reference(img, reference_image)
    if aligned_img is not None:
        aligned_images.append(aligned_img)

# Stitch the aligned images together
def stitch_images(base_img, img_to_add):
    stitched_img = np.maximum(base_img, img_to_add)
    return stitched_img

if aligned_images:
    stitched_result = aligned_images[0]
    for img in aligned_images[1:]:
        stitched_result = stitch_images(stitched_result, img)

    # Save the final stitched image to the output folder
    output_path = os.path.join(output_dir, 'stitched_output.jpg')
    cv2.imwrite(output_path, stitched_result)

    print(f"Stitched image saved as {output_path}")
else:
    print("No aligned images to stitch.")
