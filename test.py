import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from torch.utils.data import DataLoader
from train import EmotionDataset
from model import EmotionCNN

IMG_SIZE = 96
BATCH_SIZE = 64
NUM_CLASSES = 6
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_dataset = EmotionDataset(csv_file='data/annotations/test_split.csv', transform=transform_test)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =EmotionCNN(NUM_CLASSES).to(device)

label_names = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]
model.load_state_dict(torch.load("best_model.pth"))

all_predicts = []
all_labels = []
all_probs = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim = 1)
        _, predicted = torch.max(outputs, 1)
        all_predicts.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probabilities.cpu().numpy())

all_predicts = np.concatenate(all_predicts, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_probs = np.concatenate(all_probs, axis=0)

# CONFUSION MATRIX
cm = confusion_matrix(all_labels,all_predicts)
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix for Predicted Emotions')
plt.show()

y_true = label_binarize(all_labels, classes=list(range(NUM_CLASSES))) # convert the numerical labels to one-hot matrix

#  ROC PLOT
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true=y_true[:,i], y_score=all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure(figsize = (10,10))
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label='{} (AUC = {:.2f})'.format(label_names[i], roc_auc[i]))

plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multiclassifier')
plt.legend(loc="lower right")
plt.show()








