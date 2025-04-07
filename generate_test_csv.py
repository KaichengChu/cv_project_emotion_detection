import pandas as pd
import random
import os
test_dir = "data/test"
output_csv = "test_split.csv"

excluded_category = "disgusted"

dataset = []
for category in os.listdir(test_dir):
    category_path = os.path.join(test_dir, category)

    if not os.path.isdir(category_path) or category == excluded_category:
        continue

    images = os.listdir(category_path)
    images_paths = [os.path.join(category_path, img) for img in images]

    images_paths = random.sample(images_paths, 800)

    for img_path in images_paths:
        dataset.append((img_path, category))

df = pd.DataFrame(dataset, columns=["Image Path", "Category"])
df.to_csv(output_csv, index = False)
print("Job Done")