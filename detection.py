







import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO
import time
import math
from imutils import face_utils
from pygame import mixer
import imutils
import requests
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
from dotenv import load_dotenv
import os

# Fix for model loading error (1-9-25)
import torch
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('API_KEY')
SERVER_URL = os.getenv('SERVER_URL')

# Debugging: Ensure environment variables are loaded correctly
print(f"API_KEY: {API_KEY}")
print(f"SERVER_URL: {SERVER_URL}")

if not API_KEY:
    print("API_KEY not found. Make sure it's set in the .env file.")
if not SERVER_URL:

    print("SERVER_URL not found. Make sure it's set in the .env file.")

if not SERVER_URL:
    raise ValueError("SERVER_URL is not set. Check your .env file.")

# Initialize mixer for alarm
mixer.init()
mixer.music.load("music.wav")


def calculate_duration(start_time):

    if start_time is None:
        return "0 seconds"  
    # Calculate duration in seconds
    total_seconds = time.time() - start_time

    # Convert to hours, minutes, and seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format the duration
    if hours > 0:
        return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
    elif minutes > 0:
        return f"{int(minutes)} minutes, {int(seconds)} seconds"
    else:
        return f"{int(seconds)} seconds"


# Function to send detection data to the server
def send_detection(driver_id, eye_state, mouth_state, head_pose, yawning, drowsiness_status, start_time, end_time, duration):
    try:
        yawning = bool(yawning)
        drowsiness_status = "Drowsy"
        payload = {
            'driver_id': driver_id,
            'eye_state': eye_state,
            'mouth_state': mouth_state,
            'head_pose': head_pose,
            'yawning': yawning,
            'drowsiness_status': drowsiness_status,
            'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
            'duration': duration
        }
        headers = {
            'x-api-key': API_KEY,
            'Content-Type': 'application/json'
        }

        response = requests.post(SERVER_URL, json=payload, headers=headers, timeout=0.5)

        if response.status_code == 200:
            print('Data successfully sent to server.')
        else:
            print(f'Failed to send data to server. Status code: {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f'[WARNING] Failed to connect to server: {e}')


# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


# Calculate bounding box from landmarks
def calculate_bbox(landmarks):
    x_min = np.min(landmarks[:, 0])
    x_max = np.max(landmarks[:, 0])
    y_min = np.min(landmarks[:, 1])
    y_max = np.max(landmarks[:, 1])
    return x_min, y_min, x_max, y_max


# Calculate head tilt angle based on eye and nose landmarks
def head_tilt_angle(landmarks):
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    return angle


# Apply Gamma Correction
def apply_gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# Apply CLAHE for local contrast enhancement
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


# Monkey-patch for YOLOv8 model loading
import ultralytics.nn.tasks as tasks

_original_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
tasks.torch.load = patched_torch_load

# Also allow Conv & Sequential
from ultralytics.nn.modules import Conv
from torch.nn.modules.container import Sequential
torch.serialization.add_safe_globals([Conv, Sequential])

# Load YOLOv8 model
model = YOLO('model.pt')
print("Model loaded successfully!")

# Initialize dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Thresholds for EAR and MAR
thresh = 0.25
ear_thresholds = {
    'Low': 0.23,
    'Normal_Low': 0.25,
    'Normal_High': 0.30,
    'High': 0.32
}

mar_thresholds = {
    'Normal': 0.30,
    'High': 0.35
}

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize video capture
cap = cv2.VideoCapture(0)
flag = 0

# Calculate FPS
if not cap.isOpened():
    print("Failed to open video stream")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS: {fps}')

# Initialize timers
start_time_mouth = time.time()
start_time_eye = time.time()
head_tilted_start = time.time()
start_time_drowsy = None

# Define colors
COLOR_PERSON = (0, 255, 0)     # Green
COLOR_FACE = (255, 0, 0)       # Blue
COLOR_EYE = (0, 255, 255)      # Yellow
COLOR_MOUTH = (255, 0, 255)    # Magenta
previous_status = None
# 02 -09 add two line
eye_closed_start_time = None
EYE_CLOSED_DURATION = 4  # seconds (continuous eye closure required)

# --- Yawn detection variables --- paste 2 part 1 
yawn_start_time = None
yawn_count = 0
YAWN_THRESHOLD = 0.6         # MAR threshold for yawn
YAWN_MIN_DURATION = 1.0      # Seconds mouth must stay open
YAWN_SEQUENCE_LIMIT = 3      # Number of yawns before buzzer
BUZZER_DURATION = 4          # Buzzer duration in seconds
buzzer_active_until = None   # Track buzzer end time


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    drowsy = False
    current_drowsy = False

    # Lighting adjustments
    frame = apply_gamma_correction(frame, gamma=1.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)

    # Run YOLO model
    results = model(frame)

    drowsy_detected = False

    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()[:4]
        conf = detection.conf[0]
        cls = detection.cls[0]

        if conf > 0.5 and int(cls) == 0:
            class_name = model.names[int(cls)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_PERSON, 2)
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERSON, 2)

            # Extract ROI for further analysis
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            gray_roi = gray[int(y1):int(y2), int(x1):int(x2)]
            faces = detector(gray_roi)

            for face in faces:
                shape = predictor(gray_roi, face)
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                # Compute EAR, MAR, and head tilt angle
                left_eye = landmarks[lStart:lEnd]
                right_eye = landmarks[rStart:rEnd]
                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)
                mar = mouth_aspect_ratio(landmarks[48:68])
                angle = head_tilt_angle(landmarks)


                # --- Yawning Detection ---
                if mar > YAWN_THRESHOLD:
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    else:
                        if time.time() - yawn_start_time >= YAWN_MIN_DURATION:
                            cv2.putText(frame, "Yawning Detected!", (10, 180),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                            # Count the yawn and reset timer
                            yawn_count += 1
                            print(f"Yawn #{yawn_count}")
                            yawn_start_time = None  # Reset so we don’t double-count

                             # If yawns reach the limit → play buzzer
                        if yawn_count >= YAWN_SEQUENCE_LIMIT:
                            buzzer_active_until = time.time() + BUZZER_DURATION
                            mixer.music.play()  # Play buzzer once
                            yawn_count = 0      # Reset counter
                else:
                    yawn_start_time = None

# --- Control Buzzer Duration ---
                if buzzer_active_until:
                    if time.time() >= buzzer_active_until:
                        if mixer.music.get_busy():
                            mixer.music.stop()  # Stop buzzer after 4 sec
                        buzzer_active_until = None

                # Drowsiness detection logic
                # Drowsiness detection logic
                eyes_closed = (ear_left < thresh and ear_right < thresh)
                if eyes_closed:
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()  # Start timer
                    else:
                        elapsed = time.time() - eye_closed_start_time
                        if elapsed >= EYE_CLOSED_DURATION:  # Eyes closed for >= 5 sec
                            drowsy_detected = True
                            drowsy = True
                            cv2.putText(frame, 'WARNING: DRIVER IS DROWSY!', (10, 150),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if not mixer.music.get_busy():
                                mixer.music.play(-1)  # Play alarm continuously
                else:
                    eye_closed_start_time = None  # Reset timer
                    if mixer.music.get_busy():
                        mixer.music.stop()  # Stop alarm immediately

        # Update current drowsy status
        current_drowsy = drowsy

        # Send detection data if drowsiness is detected
    if drowsy_detected and eye_closed_start_time is not None:
        send_detection(
            driver_id='Driver',
            eye_state='Closed' if (ear_left < thresh or ear_right < thresh) else 'Open',
            mouth_state='Open' if mar > 0.5 else 'Closed',
            head_pose='Tilted' if abs(angle) > 15 else 'Normal',
            yawning=mar > 0.5,
            drowsiness_status=current_drowsy,
            start_time=eye_closed_start_time,
            end_time=time.time(),
            duration=calculate_duration(eye_closed_start_time)
        )


    # Display the frame
    cv2.imshow('Driver Drowsiness Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
