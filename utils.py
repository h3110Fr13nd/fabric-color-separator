import cv2
import numpy as np
import os
import argparse

def quantize_image(image_path, n_colors):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return None, None
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image from {image_path}")
        return None, None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    img_flat = img.reshape(-1, 3)  # Flatten the image to a 2D array

    # Convert to float32
    pixels = np.float32(img_flat)
    
    # Define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Apply KMeans
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8
    centers = np.uint8(centers)
    
    # Map each pixel to its corresponding centroid
    quantized = centers[labels.flatten()]
    
    # Reshape back to original image shape
    quantized_image = quantized.reshape(h, w, 3)
    
    # Also return the labels and centers for color separation
    labels_reshaped = labels.reshape(h, w)
    
    return quantized_image, (labels_reshaped, centers)

def update_color(quantized_image, labels, centers, new_centers):
    """
    Update colors in the quantized image and return new versions of all outputs.
    
    Args:
        quantized_image: The original quantized image
        labels: Color labels for each pixel
        centers: Original color centers
        new_centers: Array of new color centers
        
    Returns:
        tuple: (updated_quantized_image, updated_centers)
    """
    # Find which color indices have changed
    changed_indexes = [i for i in range(len(centers)) if not np.array_equal(centers[i], new_centers[i])]
    if not changed_indexes:
        print("No changes detected in colors.")
        return quantized_image, centers
        
    # Create a copy of the original centers
    updated_centers = centers.copy()
    
    # Update all changed colors
    for color_index in changed_indexes:
        updated_centers[color_index] = np.array(new_centers[color_index], dtype=np.uint8)
    
    # Create a new quantized image using the updated centers
    h, w, _ = quantized_image.shape
    labels_flat = labels.reshape(-1)
    updated_pixels = updated_centers[labels_flat]
    updated_quantized_image = updated_pixels.reshape(h, w, 3)
    # send the bgr centers to the color
    updated_centers = updated_centers[:, ::-1]  # Convert RGB to BGR for OpenCV
    return updated_quantized_image, updated_centers

def generate_color_variants(image_shape, labels, centers):
    """
    Generate all color variants (color images, masks, transparent images)
    without saving them to disk.
    
    Args:
        image_shape: Shape of the original image
        labels: Color labels for each pixel
        centers: Color centers
        
    Returns:
        dict: A dictionary containing all generated variants
    """
    h, w = image_shape[:2]
    results = {
        "color_images": [],
        "mask_images": [],
        "transparent_images": [],
        "color_values": centers.tolist()
    }
    
    # For each color center
    for i, color in enumerate(centers):
        # Create a mask for this color
        mask = (labels == i)
        
        # 1. Color image (white background)
        color_image = np.ones((h, w, 3), dtype=np.uint8) * 255
        color_image[mask] = color
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        results["color_images"].append(color_image_bgr)
        
        # 2. Binary mask of this color
        mask_image = np.zeros((h, w, 3), dtype=np.uint8)
        mask_image[mask] = [255, 255, 255]  # White where this color appears
        results["mask_images"].append(mask_image)
        
        # 3. Transparent background image
        transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
        transparent_image[..., 3] = 0  # Initialize all pixels as fully transparent
        
        # Convert RGB to BGR for OpenCV
        color_bgr = np.array([color[2], color[1], color[0]], dtype=np.uint8)
        
        # Set color for matched pixels
        transparent_image[mask] = np.append(color_bgr, 255)  # BGR + Alpha
        results["transparent_images"].append(transparent_image)
    
    return results

def main():
    # Use argparse for command line argument parsing
    parser = argparse.ArgumentParser(description="Quantize an image and separate colors.")
    parser.add_argument("input_path", type=str, help="Path to the input image")
    parser.add_argument("n_colors", type=int, nargs='?', default=8, help="Number of colors for quantization (default: 8)")
    
    args = parser.parse_args()
    input_path = args.input_path
    n_colors = args.n_colors
    
    # Get base filename without extension for output naming
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create output directory path
    output_dir = os.path.join("output", base_filename)
    
    print(f"Processing image: {input_path}")
    print(f"Number of colors: {n_colors}")
    print(f"Output will be saved to: {output_dir}")
    
    # Get quantized image and color data
    quantized_image, color_data = quantize_image(input_path, n_colors)
    
    if quantized_image is not None and color_data is not None:
        # Save the full quantized image
        os.makedirs(output_dir, exist_ok=True)
        quantized_path = os.path.join(output_dir, f"{base_filename}_quantized.png")
        cv2.imwrite(quantized_path, quantized_image)
        print(f"Quantized image saved to {quantized_path}")
        
        # Extract labels and centers
        labels, centers = color_data
        
        # Save individual color images
        save_individual_colors(quantized_image.shape, labels, centers, output_dir, base_filename)
    else:
        print("Failed to process the image")

if __name__ == "__main__":
    main()