import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import torch.nn.functional as F
from collections import deque
import time
import threading
import winsound
import mediapipe as mp
from math import hypot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("../models/last_epoch.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, idx_list, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idx_list]
    A = hypot(pts[1][0]-pts[5][0], pts[1][1]-pts[5][1])
    B = hypot(pts[2][0]-pts[4][0], pts[2][1]-pts[4][1])
    C = hypot(pts[0][0]-pts[3][0], pts[0][1]-pts[3][1])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

WINDOW_SIZE = 60 
prob_queue = deque(maxlen=WINDOW_SIZE)
ear_queue = deque(maxlen=WINDOW_SIZE)

DROWSY_PROB_THRESHOLD = 0.4
EAR_THRESHOLD = 0.25
DROWSY_TIME_THRESHOLD = 2.0

DROWSY_ENTER_EAR = 0.22 
DROWSY_EXIT_EAR = 0.25  

current_state = "NonDrowsy"
drowsy_start_time = None
alarm_triggered = False

def play_alarm():
    winsound.Beep(1000, 1000)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, w, h)
            ear = (left_ear + right_ear) / 2.0
            ear_queue.append(ear)

            x_min = min([int(lm.x * w) for lm in face_landmarks.landmark])
            x_max = max([int(lm.x * w) for lm in face_landmarks.landmark])
            y_min = min([int(lm.y * h) for lm in face_landmarks.landmark])
            y_max = max([int(lm.y * h) for lm in face_landmarks.landmark])
            face_img = frame[y_min:y_max, x_min:x_max]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(face_rgb)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                prob_drowsy = probs[0,0].item()
                prob_queue.append(prob_drowsy)

            avg_prob = sum(prob_queue)/len(prob_queue)
            avg_ear = sum(ear_queue)/len(ear_queue)
            current_time = time.time()

            if current_state == "NonDrowsy":
                is_drowsy_frame = (avg_prob >= DROWSY_PROB_THRESHOLD) or (avg_ear < DROWSY_ENTER_EAR)
            else:  
                is_drowsy_frame = (avg_prob >= DROWSY_PROB_THRESHOLD) or (avg_ear < DROWSY_EXIT_EAR)

            if is_drowsy_frame:
                if current_state == "NonDrowsy":
                    drowsy_start_time = current_time 
                    current_state = "Drowsy"
                    alarm_triggered = False
                elif current_state == "Drowsy":
                    elapsed = current_time - drowsy_start_time
                    if elapsed >= DROWSY_TIME_THRESHOLD and not alarm_triggered:
                        threading.Thread(target=play_alarm, daemon=True).start()
                        alarm_triggered = True
            else:
                current_state = "NonDrowsy"
                drowsy_start_time = None
                alarm_triggered = False

            label = f"{current_state} (P:{avg_prob:.2f}, EAR:{avg_ear:.2f})"

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
            cv2.putText(frame, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)

    cv2.imshow("WakeUp Drowsy Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
