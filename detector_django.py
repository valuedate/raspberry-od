import cv2
import time
import datetime
import os
import sys
import django
import numpy as np
from PIL import Image
import io
import hashlib
import logging
from threading import Thread
import signal

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
from django.db.models import Count
from django.utils import timezone
# ---------------------

from ultralytics import YOLO
import easyocr
import torch

# --- Configuration ---
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "rFERNANDES18"
CAMERA_IP = "10.0.0.115"
CHANNEL = 1
STREAM_TYPE = 1  # 1 for main stream, 2 for sub stream

RTSP_URL = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMERA_IP}:554/Streaming/channels/{CHANNEL}0{STREAM_TYPE}"

# --- Detection Settings ---
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = [0, 2]  # 0: person, 2: car
SAVE_DIR_RELATIVE = "detections"  # Relative to MEDIA_ROOT
SAVE_DIR_ABSOLUTE = os.path.join(PROJECT_DIR, "media", SAVE_DIR_RELATIVE)

# Cache directory for temporary image hashes
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Maximum number of images to keep in cache
MAX_CACHE_IMAGES = 100

# --- Duplicate Detection Settings ---
SIMILARITY_THRESHOLD = 0.85  # Higher values require more similarity to consider as duplicate
HASH_SIZE = 16  # Size of the perceptual hash (higher = more detail)
DETECTION_INTERVAL = 5  # Seconds between detections at the same location
MIN_DETECTION_INTERVAL = 1  # Minimum interval between any detections

# --- Advanced Settings ---
SAVE_FULL_FRAME = True  # Save the full frame along with the cropped detection
ENABLE_CONTINUOUS_RECORDING = False  # Record all video continuously
CONTINUOUS_RECORDING_DIR = os.path.join(PROJECT_DIR, "media", "recordings")
RECORDING_SEGMENT_MINUTES = 10
MAX_RECORDING_DAYS = 7  # Auto-delete recordings older than this many days

# --- Motion Detection ---
ENABLE_MOTION_DETECTION = True
MOTION_THRESHOLD = 25  # Lower = more sensitive
MOTION_MIN_AREA = 1000  # Minimum contour area to trigger detection
MOTION_DETECTION_INTERVAL = 0.5  # Seconds between motion detection checks

# --- Initialization ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

logger.info(f"Loading YOLO model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
model.to(device)

logger.info("Loading EasyOCR reader...")
reader = easyocr.Reader(["en"], gpu=(device=="cuda"))

os.makedirs(SAVE_DIR_ABSOLUTE, exist_ok=True)
logger.info(f"Saving detections to: {SAVE_DIR_ABSOLUTE}")

if ENABLE_CONTINUOUS_RECORDING:
    os.makedirs(CONTINUOUS_RECORDING_DIR, exist_ok=True)
    logger.info(f"Continuous recording enabled. Saving to: {CONTINUOUS_RECORDING_DIR}")

# --- Cache and State Variables ---
# Store recent image hashes to detect duplicates
recent_image_hashes = {}  # format: {label_type: [(hash, timestamp), ...]}
last_detection_time = {}  # format: {position_id: timestamp}
last_motion_check_time = 0
previous_frame = None
recording_writer = None
recording_start_time = None
running = True

def cleanup_old_recordings():
    """Delete recordings older than MAX_RECORDING_DAYS"""
    if not ENABLE_CONTINUOUS_RECORDING:
        return
        
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=MAX_RECORDING_DAYS)
    count = 0
    
    for filename in os.listdir(CONTINUOUS_RECORDING_DIR):
        filepath = os.path.join(CONTINUOUS_RECORDING_DIR, filename)
        file_time = datetime.datetime.fromtimestamp(os.path.getctime(filepath))
        
        if file_time < cutoff_date:
            os.remove(filepath)
            count += 1
    
    if count > 0:
        logger.info(f"Cleaned up {count} old recordings")

def image_hash(image):
    """Create a perceptual hash of the image for duplicate detection"""
    # Resize and convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (HASH_SIZE, HASH_SIZE))
    
    # Compute DCT (Discrete Cosine Transform)
    dct = cv2.dct(np.float32(img))
    
    # Keep only the top-left 8x8 of the DCT
    dct_low = dct[:8, :8]
    
    # Compute the median value
    median = np.median(dct_low)
    
    # Create a hash based on whether each value is above the median
    hash_value = 0
    for i in range(8):
        for j in range(8):
            if dct_low[i, j] > median:
                hash_value |= 1 << (i * 8 + j)
    
    return hash_value

def hamming_distance(hash1, hash2):
    """Calculate the Hamming distance between two hashes"""
    return bin(hash1 ^ hash2).count('1')

def is_duplicate(image, label_type):
    """Check if an image is a duplicate of a recently detected one"""
    img_hash = image_hash(image)
    
    if label_type not in recent_image_hashes:
        recent_image_hashes[label_type] = []
    
    # Compare with recent hashes of the same type
    for stored_hash, _ in recent_image_hashes[label_type]:
        distance = hamming_distance(img_hash, stored_hash)
        max_distance = HASH_SIZE * HASH_SIZE  # Maximum possible distance
        similarity = 1 - (distance / max_distance)
        
        if similarity >= SIMILARITY_THRESHOLD:
            return True
    
    # Add this hash to the recent hashes
    current_time = time.time()
    recent_image_hashes[label_type].append((img_hash, current_time))
    
    # Keep only the most recent hashes to limit memory usage
    if len(recent_image_hashes[label_type]) > MAX_CACHE_IMAGES:
        recent_image_hashes[label_type].sort(key=lambda x: x[1], reverse=True)
        recent_image_hashes[label_type] = recent_image_hashes[label_type][:MAX_CACHE_IMAGES]
    
    return False

def perform_object_detection(frame):
    results = model(frame, classes=TARGET_CLASSES, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
    return results[0]

def perform_lpr(car_image):
    try:
        gray_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive threshold for better plate reading
        gray_car_image = cv2.adaptiveThreshold(
            gray_car_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Try to enhance plate region with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        gray_car_image = cv2.morphologyEx(gray_car_image, cv2.MORPH_OPEN, kernel)
        
        ocr_results = reader.readtext(gray_car_image, detail=0, paragraph=False)
        
        # Additional filtering for license plates
        plate_texts = []
        for text in ocr_results:
            # Convert to uppercase and remove spaces
            text = text.upper().replace(' ', '')
            # Check if it's alphanumeric, has a minimum length, and contains at least one digit
            if (len(text) >= 4 and 
                text.isalnum() and 
                any(char.isdigit() for char in text)):
                plate_texts.append(text)
        
        if plate_texts:
            return max(plate_texts, key=len)
    except Exception as e:
        logger.error(f"Error during LPR: {e}")
    return None

def detect_motion(current_frame, previous_frame):
    """Detect motion between frames"""
    if previous_frame is None:
        return False, None
    
    # Convert frames to grayscale
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    frame_diff = cv2.absdiff(gray_current, gray_previous)
    
    # Apply threshold
    thresh = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate to fill in holes
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    significant_contours = [c for c in contours if cv2.contourArea(c) > MOTION_MIN_AREA]
    
    # Draw contours on a copy of the frame
    motion_overlay = current_frame.copy()
    cv2.drawContours(motion_overlay, significant_contours, -1, (0, 255, 0), 2)
    
    return len(significant_contours) > 0, motion_overlay

def start_new_recording(cap, frame):
    """Start a new video recording segment"""
    if not ENABLE_CONTINUOUS_RECORDING:
        return None, None
        
    global recording_writer, recording_start_time
    
    # Close previous writer if it exists
    if recording_writer is not None:
        recording_writer.release()
    
    # Create filename with timestamp
    timestamp = datetime.datetime.now()
    filename = f"recording_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = os.path.join(CONTINUOUS_RECORDING_DIR, filename)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recording_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    recording_start_time = timestamp
    
    # Write the first frame
    recording_writer.write(frame)
    
    logger.info(f"Started new recording segment: {filename}")
    return recording_writer, recording_start_time

def save_detection_and_metadata(frame, box, label, full_frame=None):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    detected_object_img = frame[y1:y2, x1:x2]
    
    # Check if this is a duplicate image
    if is_duplicate(detected_object_img, label):
        logger.info(f"Skipping duplicate detection: {label}")
        return False
    
    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{ts_str}.jpg"
    relative_filepath = os.path.join(SAVE_DIR_RELATIVE, filename)
    absolute_filepath = os.path.join(SAVE_DIR_ABSOLUTE, filename)
    
    try:
        # Save cropped object image
        cv2.imwrite(absolute_filepath, detected_object_img)
        
        # If enabled, save the full frame with bounding box
        if SAVE_FULL_FRAME and full_frame is not None:
            full_frame_copy = full_frame.copy()
            cv2.rectangle(full_frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(full_frame_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            full_frame_filename = f"full_{label}_{ts_str}.jpg"
            full_frame_path = os.path.join(SAVE_DIR_ABSOLUTE, full_frame_filename)
            cv2.imwrite(full_frame_path, full_frame_copy)
            
            # Store the relative path to the full frame
            full_frame_relative = os.path.join(SAVE_DIR_RELATIVE, full_frame_filename)
        else:
            full_frame_relative = None
        
        logger.info(f"Saved image: {absolute_filepath}")
        
        # Save metadata to Django database
        detection_record = Detection(
            timestamp=timestamp,
            label=label,
            image_path=relative_filepath
            # Add additional fields if you extend your model
            # full_frame_path=full_frame_relative
        )
        detection_record.save()
        logger.info(f"Saved metadata to DB for: {label}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving detection {filename}: {e}")
        return False

def generate_daily_report():
    """Generate a summary of today's detections"""
    today = timezone.now().date()
    
    # Count detections by type for today
    today_stats = Detection.objects.filter(
        timestamp__date=today
    ).values('label').annotate(count=Count('label')).order_by('-count')
    
    # Count by hour
    hourly_counts = Detection.objects.filter(
        timestamp__date=today
    ).extra({'hour': "EXTRACT(hour FROM timestamp)"}).values('hour').annotate(
        count=Count('id')).order_by('hour')
    
    # Format report
    report = f"Daily Detection Report for {today}\n"
    report += "=" * 40 + "\n\n"
    
    report += "Detections by Type:\n"
    for stat in today_stats:
        report += f"- {stat['label']}: {stat['count']}\n"
    
    report += "\nDetections by Hour:\n"
    for entry in hourly_counts:
        hour = int(entry['hour'])
        report += f"- {hour:02d}:00 - {hour+1:02d}:00: {entry['count']}\n"
    
    # Save report
    report_dir = os.path.join(PROJECT_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f"report_{today.strftime('%Y%m%d')}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Daily report generated: {report_file}")
    return report

def signal_handler(sig, frame):
    """Handle termination signals gracefully"""
    global running
    logger.info("Termination signal received. Shutting down...")
    running = False

def connect_capture_detect(rtsp_url):
    logger.info(f"Attempting to connect to: {rtsp_url.replace(CAMERA_PASSWORD, '********')}")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        logger.error("Error: Could not open video stream.")
        return
    
    logger.info("Successfully connected to the camera stream.")
    
    global previous_frame, recording_writer, recording_start_time, running
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    frame_count = 0
    processed_count = 0
    skipped_duplicates = 0
    start_time = time.time()
    
    try:
        while running:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Error: Failed to retrieve frame or stream ended. Attempting reconnect...")
                time.sleep(5)
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    logger.error("Error: Failed to reconnect.")
                    break
                else:
                    logger.info("Reconnected successfully.")
                    continue
            
            current_time = time.time()
            frame_count += 1
            
            # Continuous recording
            if ENABLE_CONTINUOUS_RECORDING:
                if recording_writer is None or recording_start_time is None:
                    recording_writer, recording_start_time = start_new_recording(cap, frame)
                else:
                    # Check if we need to start a new segment
                    segment_duration = datetime.timedelta(minutes=RECORDING_SEGMENT_MINUTES)
                    current_datetime = datetime.datetime.now()
                    if current_datetime - recording_start_time > segment_duration:
                        recording_writer, recording_start_time = start_new_recording(cap, frame)
                    else:
                        recording_writer.write(frame)
            
            # Motion detection
            motion_detected = False
            if ENABLE_MOTION_DETECTION and current_time - last_motion_check_time >= MOTION_DETECTION_INTERVAL:
                motion_detected, motion_overlay = detect_motion(frame, previous_frame)
                last_motion_check_time = current_time
                previous_frame = frame.copy()
                
                # Skip object detection if no motion is detected
                if not motion_detected and frame_count % 10 != 0:  # Still process every 10th frame regardless
                    continue
            
            # Don't process every single frame to reduce CPU load
            if frame_count % 3 != 0 and not motion_detected:
                continue
            
            # Only do full object detection if minimum interval has passed since last detection
            last_any_detection = max(last_detection_time.values()) if last_detection_time else 0
            if current_time - last_any_detection < MIN_DETECTION_INTERVAL:
                continue
            
            # Perform object detection
            results = perform_object_detection(frame)
            
            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    detected_label = model.names[class_id]
                    
                    # Calculate a position-based ID for this detection
                    cx = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    cy = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    pseudo_id = f"{detected_label}_{cx//100}_{cy//100}"
                    
                    # Check if we've recently detected something in this area
                    if current_time - last_detection_time.get(pseudo_id, 0) > DETECTION_INTERVAL:
                        logger.info(f"Detected: {detected_label} (Conf: {box.conf[0]:.2f})")
                        
                        # Default label
                        final_label = detected_label
                        
                        # Process cars differently to detect license plates
                        if detected_label == 'car':
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            car_image = frame[y1:y2, x1:x2]
                            plate_text = perform_lpr(car_image)
                            
                            if plate_text:
                                logger.info(f"  LPR Result: {plate_text}")
                                final_label = f"car_plate_{plate_text}"
                            else:
                                final_label = "car_no_plate"
                        
                        # Save the detection
                        saved = save_detection_and_metadata(frame, box, final_label, full_frame=frame)
                        
                        # Update tracking
                        if saved:
                            last_detection_time[pseudo_id] = current_time
                            processed_count += 1
                        else:
                            skipped_duplicates += 1
            
            # Print performance stats every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"Performance: {fps:.2f} FPS | Frames: {frame_count} | Processed: {processed_count} | Skipped: {skipped_duplicates}")
            
            # Generate daily report at midnight
            current_hour = datetime.datetime.now().hour
            current_minute = datetime.datetime.now().minute
            if current_hour == 0 and current_minute == 0 and frame_count % 100 == 0:
                generate_daily_report()
                cleanup_old_recordings()
            
    except KeyboardInterrupt:
        logger.info("Stream capture stopped by user.")
    finally:
        logger.info("Releasing video capture resource.")
        
        # Close video writer if open
        if recording_writer is not None:
            recording_writer.release()
        
        cap.release()
        
        # Generate final report
        generate_daily_report()