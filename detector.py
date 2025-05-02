import cv2
import time
import datetime
import os
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
# Use a smaller YOLOv8 model suitable for RPi (e.g., yolov8n.pt)
# The model will be downloaded automatically on first run
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence for detection
TARGET_CLASSES = [0, 2] # 0: person, 2: car (in COCO dataset)
SAVE_DIR = "/home/ubuntu/hikvision_detector/detections"

# --- Initialization ---
# Check if CUDA is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO model
print(f"Loading YOLO model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
model.to(device)

# Load EasyOCR reader (specify language, e.g., 'en' for English)
# This will download the language model on first run
print("Loading EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=(device=='cuda'))

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Saving detections to: {SAVE_DIR}")

def perform_object_detection(frame):
    """Performs object detection using YOLO model."""
    results = model(frame, classes=TARGET_CLASSES, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False) # verbose=False to reduce console output
    return results[0] # results is a list, take the first element

def perform_lpr(car_image):
    """Performs License Plate Recognition using EasyOCR."""
    try:
        # Convert to grayscale for potentially better OCR
        gray_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
        # Perform OCR
        ocr_results = reader.readtext(gray_car_image, detail=0, paragraph=False) # detail=0 returns only text
        # Filter results - simple alphanumeric check, length filter
        plate_texts = [text.upper() for text in ocr_results if len(text) > 4 and text.isalnum()]
        if plate_texts:
            # Simple heuristic: return the longest plausible plate text
            return max(plate_texts, key=len)
    except Exception as e:
        print(f"Error during LPR: {e}")
    return None

def save_detection(frame, box, label):
    """Saves the detected object image with timestamp and label."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    detected_object = frame[y1:y2, x1:x2]

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{ts_str}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    try:
        cv2.imwrite(filepath, detected_object)
        print(f"Saved detection: {filepath}")
        # TODO: Save metadata (filepath, timestamp, label) to Django DB in step 5
    except Exception as e:
        print(f"Error saving detection {filename}: {e}")

def connect_capture_detect(rtsp_url):
    """Connects, captures frames, performs detection and LPR."""
    print(f"Attempting to connect to: {rtsp_url.replace(CAMERA_PASSWORD, '********')}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Successfully connected to the camera stream.")

    frame_count = 0
    last_detection_time = {} # Track last detection time per object ID to avoid rapid saving
    detection_interval = 5 # Seconds between saving the same object ID

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

            # --- Object Detection ---
            results = perform_object_detection(frame)

            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    current_time = time.time()

                    # Use tracker ID if available, otherwise use class label (less reliable for distinct objects)
                    # Note: Tracking needs state, might be better handled differently for robustness
                    # Simple approach: Use class label + rough position as a pseudo-ID for throttling
                    cx = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    cy = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    pseudo_id = f"{label}_{cx//100}_{cy//100}" # Coarse grid position

                    if current_time - last_detection_time.get(pseudo_id, 0) > detection_interval:
                        print(f"Detected: {label} (Conf: {box.conf[0]:.2f})")
                        plate_text = None
                        if label == 'car':
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            car_image = frame[y1:y2, x1:x2]
                            plate_text = perform_lpr(car_image)
                            if plate_text:
                                print(f"  LPR Result: {plate_text}")
                                label = f"car_plate_{plate_text}"
                            else:
                                label = "car_no_plate"
                        elif label == 'person':
                            label = "person"

                        save_detection(frame, box, label)
                        last_detection_time[pseudo_id] = current_time

            # --- Display (Optional) ---
            # annotated_frame = results.plot() # Draw boxes on frame
            # cv2.imshow("Object Detection", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # -------------------------

            frame_count += 1
            # Optional: Add a small delay if CPU usage is too high
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("Stream capture stopped by user.")
    finally:
        print("Releasing video capture resource.")
        cap.release()
        # cv2.destroyAllWindows() # Uncomment if using imshow

if __name__ == "__main__":
    connect_capture_detect(RTSP_URL)

