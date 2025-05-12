import gradio as gr
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def enhance_and_compare(image, clahe_type, clip_limit, tile_grid_size, gamma_value, denoise_strength, sharpen):
    # Check if image exists
    if image is None:
        return [np.zeros((300, 300, 3), dtype=np.uint8), 
                np.zeros((300, 300, 3), dtype=np.uint8), 
                "Error: No image uploaded"]
    
    # Ensure image is in BGR format
    if len(image.shape) != 3 or image.shape[2] != 3:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        except:
            return [np.zeros((300, 300, 3), dtype=np.uint8), 
                    np.zeros((300, 300, 3), dtype=np.uint8), 
                    "Error: Could not convert image to BGR"]
    
    # Image enhancement
    try:
        # Convert image to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(image_rgb)
        
        # Apply CLAHE
        if clahe_type == "RGB":
            # CLAHE on each RGB channel separately
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            r = clahe.apply(r)
            g = clahe.apply(g)
            b = clahe.apply(b)
        elif clahe_type == "YUV":
            # Convert image to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            
            # CLAHE on luminance channel only
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            y = clahe.apply(y)
            
            # Merge channels
            yuv = cv2.merge([y, u, v])
            image_rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            r, g, b = cv2.split(image_rgb)
        else:
            # No CLAHE
            pass
        
        # Gamma correction
        r = np.power(r / 255.0, gamma_value) * 255
        g = np.power(g / 255.0, gamma_value) * 255
        b = np.power(b / 255.0, gamma_value) * 255
        
        # Ensure values are in the correct range
        r = np.clip(r, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        
        # Merge channels
        enhanced_image = cv2.merge([r, g, b])
        enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        
        # Denoising
        if denoise_strength > 0:
            enhanced_bgr = cv2.fastNlMeansDenoisingColored(
                enhanced_bgr, None,
                h=denoise_strength, hColor=denoise_strength,
                templateWindowSize=7, searchWindowSize=21
            )
        
        # Sharpening
        if sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced_bgr = cv2.filter2D(enhanced_bgr, -1, kernel)
    
    except Exception as e:
        return [np.zeros((300, 300, 3), dtype=np.uint8), 
                np.zeros((300, 300, 3), dtype=np.uint8), 
                f"Error during image enhancement: {str(e)}"]
    
    # Try to find ground truth image
    try:
        # Get the current example index
        current_example = None
        for example in examples:
            if isinstance(example[0], str) and os.path.exists(example[0]):
                example_img = cv2.imread(example[0])
                if example_img is not None and np.array_equal(image, example_img):
                    current_example = example
                    break
        
        if current_example is not None:
            input_filename = os.path.basename(current_example[0])
        else:
            # If not found in examples, try to get filename from image path if it exists
            if hasattr(image, 'filename') and image.filename:
                input_filename = os.path.basename(image.filename)
            else:
                input_filename = "unknown.png"
        
        # Remove file extension for matching
        base_filename = os.path.splitext(input_filename)[0]
        
        # List of possible ground truth image paths
        possible_paths = [
            f"data/high/{base_filename}.png",
            f"data/high/{base_filename}.jpg",
            f"data/high/{base_filename}.jpeg"
        ]
        
        ground_truth = None
        for path in possible_paths:
            if os.path.exists(path):
                ground_truth = cv2.imread(path)
                if ground_truth is not None:
                    # Resize ground truth image to match enhanced image
                    ground_truth = cv2.resize(ground_truth, (enhanced_bgr.shape[1], enhanced_bgr.shape[0]))
                    break
        
        # If no ground truth image found
        if ground_truth is None:
            ground_truth = np.zeros_like(enhanced_bgr)
            error_msg = f"No ground truth image found for {input_filename}. Using a blank image."
        else:
            error_msg = ""
    
    except Exception as e:
        ground_truth = np.zeros_like(enhanced_bgr)
        error_msg = f"Error loading ground truth image: {str(e)}"
    
    # Calculate SSIM
    try:
        enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
        similarity = ssim(enhanced_gray, gt_gray)
        
        # Prepare similarity text
        similarity_text = f"SSIM Similarity: {similarity:.4f}"
        if error_msg:
            similarity_text += f"\n\nERROR: {error_msg}"
    
    except Exception as e:
        similarity_text = f"Error calculating SSIM: {str(e)}"
        similarity = 0.0
    
    return [enhanced_bgr, ground_truth, similarity_text]

# Gradio Interface Setup
examples = [
    [os.path.join("data", "low", "102.png"), "RGB", 2.0, 8, 1.0, 10, True],
    [os.path.join("data", "low", "13.png"), "YUV", 3.0, 8, 0.8, 15, False],
    [os.path.join("data", "low", "18.png"), "RGB", 1.5, 8, 1.2, 20, True],
    [os.path.join("data", "low", "2.png"), "YUV", 2.5, 8, 0.9, 12, True],
    [os.path.join("data", "low", "21.png"), "RGB", 2.2, 8, 1.1, 8, False],
    [os.path.join("data", "low", "39.png"), "YUV", 1.8, 8, 1.3, 18, True],
    [os.path.join("data", "low", "43.png"), "RGB", 2.7, 8, 0.7, 14, False],
    [os.path.join("data", "low", "48.png"), "YUV", 1.6, 8, 1.4, 22, True],
    [os.path.join("data", "low", "5.png"), "RGB", 2.3, 8, 0.6, 16, False],
    [os.path.join("data", "low", "52.png"), "YUV", 1.9, 8, 1.2, 11, True],
    [os.path.join("data", "low", "54.png"), "RGB", 2.6, 8, 0.5, 19, False],
    [os.path.join("data", "low", "57.png"), "YUV", 2.1, 8, 1.5, 13, True],
    [os.path.join("data", "low", "60.png"), "RGB", 1.7, 8, 0.4, 17, False],
    [os.path.join("data", "low", "75.png"), "YUV", 2.4, 8, 1.6, 21, True],
    [os.path.join("data", "low", "83.png"), "RGB", 2.8, 8, 0.3, 23, False]
]

gr.Interface(
    fn=enhance_and_compare,
    inputs=[
        gr.Image(type="numpy", label="Low-Quality Input Image"),
        gr.Dropdown(choices=["No CLAHE", "RGB", "YUV"], label="CLAHE Type", value="No CLAHE"),
        gr.Slider(minimum=1.0, maximum=10.0, value=2.0, label="Clip Limit"),
        gr.Slider(minimum=2, maximum=16, value=8, step=2, label="Tile Grid Size"),
        gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.1, label="Gamma Correction"),
        gr.Slider(minimum=0, maximum=30, value=10, label="Denoising Strength"),
        gr.Checkbox(label="Sharpen", value=True)
    ],
    outputs=[
        gr.Image(type="numpy", label="Enhanced Image"),
        gr.Image(type="numpy", label="Ground Truth Image"),
        gr.Text(label="Similarity Score (SSIM)")
    ],
    title="Image Enhancement & Quality Comparison",
    description="Upload a low-quality image and compare it with the ground truth using SSIM.",
    examples=examples
).launch()