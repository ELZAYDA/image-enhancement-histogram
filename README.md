
# Image Enhancement Project

## Project Purpose and Overview

This project focuses on enhancing low-quality images using advanced image processing techniques such as CLAHE (Contrast Limited Adaptive Histogram Equalization), Gamma Correction, and Denoising. The goal is to improve the quality of images and evaluate their performance by comparing the enhanced images to the corresponding high-quality ground truth images. The project also calculates and logs the Structural Similarity Index (SSIM) score to assess the quality of enhancement.

## Setup Instructions and Dependencies

### Prerequisites
Ensure that you have the following installed on your system:
- Python 3.x
- OpenCV
- NumPy
- scikit-image
- Matplotlib

### Installation
To set up the project, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/image-enhancement-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd image-enhancement-project
   ```
3. Install the necessary dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

### Directory Structure
The project follows this directory structure:
```
image-enhancement-project/
│
├── data/                  # Contains input low and high images (subdirectories: low, high)
├── results/               # Results directory for saving enhanced images, plots, and metrics
│   ├── enhanced/          # Enhanced images
│   ├── plots/             # Plots for comparisons and histograms
│   ├── metrics/           # Logs of SSIM scores and metrics
├── requirements.txt       # List of dependencies
├── script.py              # Main script for image processing
└── README.md              # Project overview and documentation
```

## Usage Examples

### Input Images
This project expects two sets of images:
- **Low-quality images** (stored in the `data/low` folder).
- **High-quality ground truth images** (stored in the `data/high` folder).

Ensure that the number of low and high images are the same, and they are named similarly.

### Example Command
To run the image enhancement and save results, execute the following Python script:
```bash
python script.py
```

### Output
The output will be saved in the `results/` folder:
- **Enhanced Images** will be saved in the `results/enhanced` folder.
- **Comparison Plots** (Original, Enhanced, Ground Truth) will be saved in the `results/plots` folder.
- **SSIM Scores** will be logged in the `results/metrics/metrics.csv` file.
- **Histograms** of original, enhanced, and ground truth images will be saved in the `results/plots` folder.

### Example of Input and Output:

#### Input (Low-Quality Image):
![Low-Quality](data/low/sample_low_image.jpg)

#### Output (Enhanced Image):
![Enhanced](results/enhanced/sample_enhanced_image.jpg)

#### Ground Truth (High-Quality Image):
![Ground Truth](data/high/sample_high_image.jpg)

### SSIM Score:
The Structural Similarity Index (SSIM) is calculated to measure the similarity between the enhanced and ground truth images. The SSIM score is logged for each image pair.

## Results and Performance Metrics

The project evaluates the enhancement quality using the **Structural Similarity Index (SSIM)** between the enhanced image and the corresponding high-quality ground truth image. The SSIM scores are saved in a CSV file for further analysis.

Example CSV log entry:
```
timestamp,image_name,ssim
2025-05-12 12:34:56,sample_low_image,0.85
```

## Contributing
If you'd like to contribute to the project, feel free to fork the repository and submit pull requests. Any suggestions for improving the image enhancement methods or performance metrics are welcome!

## Requirements

Before running the script, make sure to install the necessary dependencies. You can install them using `pip` by running the following command:

```bash
pip install -r requirements.txt
