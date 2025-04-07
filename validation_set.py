import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full training dataset
df = pd.read_csv("data/annotations/train_labels.csv")

# Perform 90:10 split
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["Category"], random_state=32)

# Save the split datasets
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)

print(f"Training set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")
