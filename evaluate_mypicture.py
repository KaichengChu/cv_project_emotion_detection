import torch
import torchvision.transforms as transforms
from PIL import Image
from model import EmotionCNN  # Import the model class

# Define the transformation (same as used during training)
IMG_SIZE = 96
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# Label mapping
label_to_idx = {"angry": 0, "fearful": 1, "happy": 2, "neutral": 3, "sad": 4, "surprised": 5}
idx_to_label = {v: k for k, v in label_to_idx.items()}

# Load the model
num_classes = len(label_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()


# Function to predict emotion
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    return idx_to_label[predicted_class]


# Example usage
if __name__ == "__main__":
    image_path = "data/MyEmotions/ckc_unknown.png"  # Change this to your image path
    predicted_emotion = predict(image_path)
    print(f"Predicted Emotion: {predicted_emotion}")
