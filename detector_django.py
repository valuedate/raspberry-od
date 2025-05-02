import cv2
import time
import datetime
import os
import sys
import django

# --- Django Setup ---
# Add the project directory to the Python path
PROJECT_DIR = "/home/ubuntu/hikvision_detector"
sys.path.append(PROJECT_DIR)
# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
# Initialize Django
django.setup()

# Import the model after Django setup
from viewer.models import Detection
# ---------------------

from ultralytics import YOLO
import easyocr
import torch

# --- Configuration ---
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "123456"
CAMERA_IP = "10.0.0.115"
CHANNEL = 1
STREAM_TYPE = 1  # 1 for main stream, 2 for sub stream

RTSP_URL = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMERA_IP}:554/Streaming/channels/{CHANNEL}0{STREAM_TYPE}"

# --- Detection Settings ---
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = [0, 2] # 0: person, 2: car
SAVE_DIR_RELATIVE = "detections" # Relative to MEDIA_ROOT
SAVE_DIR_ABSOLUTE = os.path.join(PROJECT_DIR, "media", SAVE_DIR_RELATIVE)

# --- Initialization ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print(f"Loading YOLO model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
model.to(device)

print("Loading EasyOCR reader...")
reader = easyocr.Reader(["en"], gpu=(device=="cuda"))

os.makedirs(SAVE_DIR_ABSOLUTE, exist_ok=True)
print(f"Saving detections to: {SAVE_DIR_ABSOLUTE}")

def perform_object_detection(frame):
    results = model(frame, classes=TARGET_CLASSES, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
    return results[0]

def perform_lpr(car_image):
    try:
        gray_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray_car_image, detail=0, paragraph=False)
        plate_texts = [text.upper() for text in ocr_results if len(text) > 4 and text.isalnum()]
        if plate_texts:
            return max(plate_texts, key=len)
    except Exception as e:
        print(f"Error during LPR: {e}")
    return None

def save_detection_and_metadata(frame, box, label):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    detected_object_img = frame[y1:y2, x1:x2]

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{ts_str}.jpg"
    relative_filepath = os.path.join(SAVE_DIR_RELATIVE, filename)
    absolute_filepath = os.path.join(SAVE_DIR_ABSOLUTE, filename)

    try:
        # Save image file
        cv2.imwrite(absolute_filepath, detected_object_img)
        print(f"Saved image: {absolute_filepath}")

        # Save metadata to Django database
        detection_record = Detection(
            timestamp=timestamp,
            label=label,
            image_path=relative_filepath # Store relative path for Django media handling
        )
        detection_record.save()
        print(f"Saved metadata to DB for: {label}")

    except Exception as e:
        print(f"Error saving detection {filename}: {e}")

def connect_capture_detect(rtsp_url):
    print(f"Attempting to connect to: {rtsp_url.replace(CAMERA_PASSWORD, '********')}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Successfully connected to the camera stream.")

    frame_count = 0
    last_detection_time = {}
    detection_interval = 5 # Seconds

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to retrieve frame or stream ended. Attempting reconnect...")
                time.sleep(5)
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    print("Error: Failed to reconnect.")
                    break
                else:
                    print("Reconnected successfully.")
                    continue

            results = perform_object_detection(frame)

            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    detected_label = model.names[class_id]
                    current_time = time.time()

                    cx = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    cy = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    pseudo_id = f"{detected_label}_{cx//100}_{cy//100}"

                    if current_time - last_detection_time.get(pseudo_id, 0) > detection_interval:
                        print(f"Detected: {detected_label} (Conf: {box.conf[0]:.2f})")
                        final_label = detected_label # Default label
                        plate_text = None
                        if detected_label == 'car':
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            car_image = frame[y1:y2, x1:x2]
                            plate_text = perform_lpr(car_image)
                            if plate_text:
                                print(f"  LPR Result: {plate_text}")
                                final_label = f"car_plate_{plate_text}"
                            else:
                                final_label = "car_no_plate"
                        elif detected_label == 'person':
                            final_label = "person"

                        save_detection_and_metadata(frame, box, final_label)
                        last_detection_time[pseudo_id] = current_time

            frame_count += 1
            # time.sleep(0.01) # Optional delay

    except KeyboardInterrupt:
        print("Stream capture stopped by user.")
    finally:
        print("Releasing video capture resource.")
        cap.release()

if __name__ == "__main__":
    # Note: This script should ideally be run as a Django management command
    # or a separate process managed by systemd/supervisor on the RPi.
    # Running it directly like this is mainly for testing the core logic.
    connect_capture_detect(RTSP_URL)

