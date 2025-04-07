import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import EmotionCNN

new_path = r"C:\Users\chuka\Desktop\CS4501\project"
os.chdir(new_path)


# Hyperparameters
BATCH_SIZE = 64
FINETUNE_EPOCHS = 10
FINETUNE_LR = 0.0015
IMG_SIZE = 96



#Tramsformation for training data
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Transformation for testing data
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# Custome dataset

label_to_idx = {"angry":0, "fearful":1, "happy":2, "neutral":3, "sad":4, "surprised":5 }
class EmotionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # read the data file. Format: filepath : emotion category
        self.data = pd.read_csv(csv_file, header=0)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get the image name
        img_path = self.data.iloc[idx, 0]
        # get the emotion category label
        label_str = self.data.iloc[idx, 1]
        # load picture
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        # convert the label to numerical value
        label = label_to_idx[label_str]
        label = torch.tensor(label, dtype=torch.long)
        return image, label

train_dataset = EmotionDataset("data/annotations/train_split.csv", transform=transform_train)
val_dataset = EmotionDataset("data/annotations/val_split.csv", transform= transform_test)
test_dataset = EmotionDataset("data/annotations/test_split.csv", transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.data["Category"].unique())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =EmotionCNN(num_classes).to(device)


# Define loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(model.parameters(),
                      lr=FINETUNE_LR,
                      momentum=0.9,
                      weight_decay=0.0001)

def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(loader), correct / total

best_val_loss = float('inf')
patience = 3
counter = 0
delta = 0.01

for epoch in range(FINETUNE_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)
    print(
        f"Epoch {epoch + 1}/{FINETUNE_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

    # Early stopping logic
    if best_val_loss - val_loss > delta:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# -------------------------------
# Final Evaluation on Test Set
# -------------------------------
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
_, test_acc = evaluate(model, test_loader)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

