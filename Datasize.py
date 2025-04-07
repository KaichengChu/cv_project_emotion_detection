from PIL import Image
import os

# Path to a sample image
sample_image_path = "data/MyEmotions/yxp_surprised.png"  # Change to an actual image path

# Open the image
with Image.open(sample_image_path) as img:
    print(f"Image size: {img.size}")  # Outputs (width, height)
