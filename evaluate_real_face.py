import cv2
import time
import os
import torch
from torchvision import transforms
from PIL import Image
from model import EmotionCNN

# Set up the working directory
new_path = r"C:\Users\chuka\Desktop\CS4501\project\real_time_face"
os.chdir(new_path)

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
webcam = cv2.VideoCapture(0)

# Initialize snapshot variables
snapshot_count = 0
last_saved_time = 0
save_interval = 8

# Set up the emotion recognition model
NUM_CLASSES = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(r"C:\Users\chuka\Desktop\CS4501\project\best_model.pth", map_location=device))
model.eval()

# Define the label names for emotions
label_names = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]

# Define the transform used during training/testing
IMG_SIZE = 96
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    current_time = time.time()

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Crop the face region from the frame
        face_roi = frame[y:y + h, x:x + w]

        # Optionally save the face snapshot (only every save_interval seconds)
        if current_time - last_saved_time >= save_interval:
            snapshot_filename = os.path.join(new_path, f"face_snapshot_{snapshot_count}.png")
            cv2.imwrite(snapshot_filename, face_roi)
            print(f"Snapshot saved as {snapshot_filename}")
            snapshot_count += 1
            last_saved_time = current_time

        # Convert the cropped face to RGB and then to PIL Image (required by transforms)
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        # Preprocess the image
        input_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Run the model to predict the emotion
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, 1)
            emotion = label_names[predicted.item()]

        # Overlay the predicted emotion on the frame near the face rectangle
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36, 255, 12), 2)

    cv2.imshow("Face Detection with Emotion Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
