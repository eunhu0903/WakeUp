import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from prepare_data import train_dataset, val_dataset, test_dataset
from collections import Counter
import random

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("클래스 인덱스:", train_dataset.class_to_idx)

    # --------------------------
    # 데이터 균형 확인 & 증강
    # --------------------------
    counter = Counter([label for _, label in train_dataset.samples])
    print("학습 데이터 클래스별 개수:")
    for cls_idx, count in counter.items():
        print(f"{train_dataset.classes[cls_idx]}: {count} images")

    drowsy_count = counter[train_dataset.class_to_idx['Drowsy']]
    nondrowsy_count = counter[train_dataset.class_to_idx['NonDrowsy']]
    target_ratio = 1.0  # Drowsy와 균형 맞추기

    if nondrowsy_count < drowsy_count * target_ratio:
        print("NonDrowsy 데이터 증강 필요")
        needed = drowsy_count - nondrowsy_count
        print(f"증강 필요 이미지 수: {needed}")

        nondrowsy_idx = [i for i, (_, label) in enumerate(train_dataset.samples)
                         if label == train_dataset.class_to_idx['NonDrowsy']]
        augmented_samples = []

        for i in range(needed):
            idx = random.choice(nondrowsy_idx)
            path, label = train_dataset.samples[idx]
            augmented_samples.append((path, label))  # transform에서 증강 적용

        train_dataset.samples.extend(augmented_samples)
        print(f"총 train 샘플 수: {len(train_dataset.samples)}")
    else:
        print("데이터 균형 이미 충분함")

    # --------------------------
    # transform 적용
    # --------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    for ds in [train_dataset, val_dataset, test_dataset]:
        ds.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # --------------------------
    # 모델 로드 & fine-tuning 설정
    # --------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    num_epochs = 20
    best_val_acc = 0.0
    os.makedirs("../models", exist_ok=True)

    # --------------------------
    # 학습 루프
    # --------------------------
    total_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 배치 단위 진행률 & 남은 시간
            if (i+1) % 50 == 0 or (i+1) == len(train_loader):
                elapsed = time.time() - epoch_start_time
                batches_done = i + 1
                batches_total = len(train_loader)
                est_total = elapsed / batches_done * batches_total
                est_remaining = est_total - elapsed
                print(f"Batch [{batches_done}/{batches_total}] - "
                      f"Elapsed: {elapsed:.1f}s, Est. remaining for epoch: {est_remaining:.1f}s")

        train_loss = running_loss / total
        train_acc = correct / total

        # --------------------------
        # 검증
        # --------------------------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = val_correct / val_total

        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - total_start_time
        est_total_time = total_elapsed / (epoch+1) * num_epochs
        est_remaining_total = est_total_time - total_elapsed

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
              f"Epoch Time: {epoch_elapsed:.1f}s, Est. remaining total: {est_remaining_total/60:.1f} min")

        # 모델 저장
        torch.save(model.state_dict(), f"../models/last_epoch.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../models/best_model.pth")
            print("✅ Best model saved.")

    # --------------------------
    # 테스트
    # --------------------------
    model.load_state_dict(torch.load("../models/best_model.pth"))
    model.eval()
    test_correct, test_total = 0, 0
    class_correct, class_total = [0, 0], [0, 0]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    test_acc = test_correct / test_total
    print(f"\n전체 Test Accuracy: {test_acc:.4f}")
    print(f"Drowsy 정확도: {class_correct[0]/class_total[0]:.4f}")
    print(f"NonDrowsy 정확도: {class_correct[1]/class_total[1]:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
