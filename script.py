import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import csv
from datetime import datetime

def setup_directories():
    """Create necessary directories if they do not exist."""
    os.makedirs('results/equalized', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

def load_image_pairs(low_dir='data/low', high_dir='data/high'):
    """Load pairs of low-quality and high-quality images."""
    low_images = sorted([f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    high_images = sorted([f for f in os.listdir(high_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    pairs = []
    for low_img, high_img in zip(low_images, high_images):
        low_path = os.path.join(low_dir, low_img)
        high_path = os.path.join(high_dir, high_img)
        pairs.append((low_path, high_path))

    return pairs

def enhance_image(image):
    """Enhance the image using histogram equalization."""
    enhanced = cv2.equalizeHist(image)
    return enhanced

def save_comparison_plot(original, enhanced, ground_truth, ssim_score, filename):
    """Save a comparison plot of the original, enhanced, and ground truth images."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title(f'Enhanced\nSSIM: {ssim_score:.4f}')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plot_path = os.path.join('results', 'plots', f'{filename}_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

def log_metrics(filename, ssim_score, log_file='results/metrics/metrics.csv'):
    """Log SSIM metrics to a CSV file."""
    file_exists = os.path.isfile(log_file)

    with open(log_file, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'image_name', 'ssim']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_name': filename,
            'ssim': ssim_score
        })

def process_images():
    """Process the images, enhance them, and log the results."""
    setup_directories()
    image_pairs = load_image_pairs()

    print(f"Found {len(image_pairs)} image pairs to process")

    for low_path, high_path in image_pairs:
        original = cv2.imread(low_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.imread(high_path, cv2.IMREAD_GRAYSCALE)

        if original is None or ground_truth is None:
            print(f"Warning: Could not load image pair {low_path} and {high_path}")
            continue

        enhanced = enhance_image(original)
        ssim_score = ssim(enhanced, ground_truth, data_range=255)
        filename = os.path.splitext(os.path.basename(low_path))[0]

        # Save the enhanced image
        enhanced_path = os.path.join('results', 'equalized', f'{filename}_equalized.png')
        cv2.imwrite(enhanced_path, enhanced)

        # Save comparison plot
        save_comparison_plot(original, enhanced, ground_truth, ssim_score, filename)

        # Log SSIM score
        log_metrics(filename, ssim_score)

        print(f"Processed {filename}: SSIM = {ssim_score:.4f}")

if __name__ == '__main__':
    process_images()
