import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from prepare_data import train_dataset, val_dataset, test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("클래스 인덱스:", train_dataset.class_to_idx)

# --------------------------
# 데이터 전처리 (test dataset)
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
test_dataset.transform = transform
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# --------------------------
# 모델 로드
# --------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# 최신 모델 경로 (last_epoch 기준)
model_path = "../models/best_model.pth"
if not os.path.exists(model_path):
    model_path = "../models/wakeup_resnet.pth"

print(f"모델 로드: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --------------------------
# 테스트 정확도 계산
# --------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 정확도
accuracy = (all_preds == all_labels).mean()
print(f"\n전체 Test Accuracy: {accuracy:.4f}")

# 클래스별 성능
print("\n[클래스별 성능]")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes, digits=4))

# 혼동행렬
print("\n[혼동 행렬]")
print(confusion_matrix(all_labels, all_preds))
