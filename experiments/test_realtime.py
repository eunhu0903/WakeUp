import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import torch.nn.functional as F
from collections import deque
import time
import threading
import winsound  # Windows에서 소리 재생

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 모델 로드
# -------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("../models/last_epoch.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------------------
# 이미지 전처리
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------------
# 얼굴 검출기
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------------------
# 웹캠 실행
# -------------------------------
cap = cv2.VideoCapture(0)

# 최근 프레임 Drowsy 확률 저장
prob_queue = deque(maxlen=30)  # 약 1초치 프레임

# Drowsy 판단 기준
DROWSY_PROB_THRESHOLD = 0.5  # 평균 확률 0.5 이상이면 Drowsy
ALARM_DURATION = 1000  # 소리 길이(ms)
ALARM_FREQ = 1000     # 소리 주파수(Hz)
DROWSY_TIME_THRESHOLD = 3.0  # 초 단위, 3초 이상일 때 알람

# Drowsy 상태 지속 시간 체크
drowsy_start_time = None
alarm_triggered = False  # 알람 중복 방지

# -------------------------------
# 알람 함수 (쓰레드 사용)
# -------------------------------
def play_alarm():
    winsound.Beep(ALARM_FREQ, ALARM_DURATION)

# -------------------------------
# 메인 루프
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(face_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            prob_drowsy = probs[0, 0].item()
            prob_queue.append(prob_drowsy)

        # 최근 프레임 평균 확률 계산
        avg_prob = sum(prob_queue) / len(prob_queue)
        current_time = time.time()

        if avg_prob >= DROWSY_PROB_THRESHOLD:
            label = f"Drowsy ({avg_prob:.2f})"
            if drowsy_start_time is None:
                drowsy_start_time = current_time
                alarm_triggered = False  # 새로운 Drowsy 시작
            elapsed = current_time - drowsy_start_time

            # 3초 이상 지속 시 알람
            if elapsed >= DROWSY_TIME_THRESHOLD and not alarm_triggered:
                threading.Thread(target=play_alarm, daemon=True).start()
                alarm_triggered = True
        else:
            label = f"NonDrowsy ({1-avg_prob:.2f})"
            drowsy_start_time = None
            alarm_triggered = False

        # 얼굴 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("WakeUp Test (Face Only)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
