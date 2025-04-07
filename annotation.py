import os
import pandas as pd
import random

train_dir = "data/train"
output_csv = "train_labels.csv"

exclude_category = "disgusted"
down_sample_category = "happy"
down_sample_target = 4500

dataset = []

random.seed(32)

for category in os.listdir(train_dir):
    category_path = os.path.join(train_dir, category)

    if not os.path.isdir(category_path) or category == exclude_category:
        continue

    images = os.listdir(category_path)
    image_paths = [os.path.join(category_path, img) for img in images]

    if category == down_sample_category and len(image_paths) > down_sample_target:
        image_paths = random.sample(image_paths, down_sample_target)

    for img_path in image_paths:
        dataset.append((img_path, category))

df = pd.DataFrame(dataset, columns=["Image Path", "Category"])
df.to_csv(output_csv, index=False)

print("Annotation saved!")
