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
import traceback

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
PROJECT_DIR = "/home/admin/raspberry-od"
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
MAX_CACHE_IMAGES = 250  # Increased from 100 to 250

# --- Duplicate Detection Settings ---
SIMILARITY_THRESHOLD = 0.90  # Increased from 0.85 to 0.90
HASH_SIZE = 32  # Increased from 16 to 32
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

# --- License Plate Recognition Settings ---
LPR_MIN_PLATE_ASPECT_RATIO = 1.5
LPR_MAX_PLATE_ASPECT_RATIO = 5.0
LPR_MIN_PLATE_AREA_RATIO = 0.005  # Min plate area relative to car area
LPR_MAX_PLATE_AREA_RATIO = 0.15   # Max plate area relative to car area

# --- Database Cleanup Settings ---
DB_CLEANUP_ENABLED = True
DB_CLEANUP_DAYS = 30  # Remove detections older than this many days
DB_CLEANUP_INTERVAL_HOURS = 24  # Run cleanup once per day

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
last_db_cleanup_time = 0
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
            try:
                os.remove(filepath)
                count += 1
            except Exception as e:
                logger.error(f"Error removing old recording {filepath}: {e}")
    
    if count > 0:
        logger.info(f"Cleaned up {count} old recordings")

def cleanup_database():
    """Remove old records from the database"""
    if not DB_CLEANUP_ENABLED:
        return
        
    try:
        cutoff_date = timezone.now() - datetime.timedelta(days=DB_CLEANUP_DAYS)
        old_detections = Detection.objects.filter(timestamp__lt=cutoff_date)
        count = old_detections.count()
        
        # Delete associated files first
        for detection in old_detections:
            try:
                file_path = os.path.join(PROJECT_DIR, "media", detection.image_path)
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing file for detection {detection.id}: {e}")
        
        # Then delete the database records
        old_detections.delete()
        
        logger.info(f"Database cleanup: removed {count} old detection records")
    except Exception as e:
        logger.error(f"Error during database cleanup: {e}")

def image_hash(image):
    """Create an improved perceptual hash of the image for duplicate detection"""
    if image is None or image.size == 0:
        return 0
    
    # Resize and convert to grayscale
    try:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0
        
    img = cv2.resize(img, (HASH_SIZE, HASH_SIZE))
    
    # Apply CLAHE to normalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Compute DCT (Discrete Cosine Transform)
    dct = cv2.dct(np.float32(img))
    
    # Keep only the top-left (HASH_SIZE/4)x(HASH_SIZE/4) of the DCT for a more robust hash
    dct_low = dct[:HASH_SIZE//4, :HASH_SIZE//4]
    
    # Compute the median value
    median = np.median(dct_low)
    
    # Create a hash based on whether each value is above the median
    hash_value = 0
    bit_count = 0
    for i in range(dct_low.shape[0]):
        for j in range(dct_low.shape[1]):
            bit_count += 1
            if dct_low[i, j] > median:
                hash_value |= 1 << bit_count
    
    return hash_value

def hamming_distance(hash1, hash2):
    """Calculate the Hamming distance between two hashes"""
    return bin(hash1 ^ hash2).count('1')

def is_duplicate(image, label_type):
    """Check if an image is a duplicate of a recently detected one"""
    if image is None or image.size == 0:
        return True  # Consider empty images as duplicates
    
    img_hash = image_hash(image)
    
    if img_hash == 0:  # Invalid hash
        return True
    
    if label_type not in recent_image_hashes:
        recent_image_hashes[label_type] = []
    
    # Compare with recent hashes of the same type
    max_hash_distance = (HASH_SIZE // 4) * (HASH_SIZE // 4)  # Maximum possible distance
    
    for stored_hash, _ in recent_image_hashes[label_type]:
        distance = hamming_distance(img_hash, stored_hash)
        similarity = 1 - (distance / max_hash_distance)
        
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

def enhance_image(image):
    """Apply image enhancement techniques"""
    if image is None or image.size == 0:
        return None
        
    try:
        # Convert to appropriate size
        height, width = image.shape[:2]
        # Don't enlarge small images too much to avoid artifacts
        scale = min(2.0, max(1.0, 800 / max(width, height)))
        if scale > 1.0:
            image = cv2.resize(image, (int(width * scale), int(height * scale)), 
                              interpolation=cv2.INTER_CUBIC)
        
        # Apply mild denoise (careful not to lose details)
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        
        # Enhance contrast using CLAHE on the L channel (in LAB color space)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply mild sharpening
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return image  # Return original if enhancement fails

def perform_object_detection(frame):
    """Perform object detection with YOLO"""
    try:
        results = model(frame, classes=TARGET_CLASSES, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
        return results[0]
    except Exception as e:
        logger.error(f"Error during object detection: {e}")
        return None

def detect_license_plate_region(car_image):
    """Detect the license plate region in a car image"""
    if car_image is None or car_image.size == 0:
        return None
        
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while removing noise
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges
        edged = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        car_area = car_image.shape[0] * car_image.shape[1]
        
        # Try to find a rectangle that might be the license plate
        for c in contours:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            
            # If our contour has four points, it could be a license plate
            if len(approx) >= 4 and len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check if dimensions are reasonable for a plate
                aspect_ratio = float(w) / h
                area_ratio = (w * h) / car_area
                
                if (LPR_MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= LPR_MAX_PLATE_ASPECT_RATIO and
                    LPR_MIN_PLATE_AREA_RATIO <= area_ratio <= LPR_MAX_PLATE_AREA_RATIO):
                    # Add a small margin around the plate
                    margin_x = int(w * 0.05)
                    margin_y = int(h * 0.1)
                    
                    # Ensure we stay within image boundaries
                    x_start = max(0, x - margin_x)
                    y_start = max(0, y - margin_y)
                    x_end = min(car_image.shape[1], x + w + margin_x)
                    y_end = min(car_image.shape[0], y + h + margin_y)
                    
                    plate_region = car_image[y_start:y_end, x_start:x_end]
                    return plate_region
        
        return None
    except Exception as e:
        logger.error(f"Error detecting license plate region: {e}")
        traceback.print_exc()
        return None

def perform_lpr(car_image):
    """Enhanced license plate recognition with multiple preprocessing approaches"""
    if car_image is None or car_image.size == 0:
        return None
        
    try:
        # Enhance the entire car image first
        enhanced_car = enhance_image(car_image)
        
        # Try to detect the license plate region
        plate_region = detect_license_plate_region(enhanced_car)
        
        # Initialize a list to store OCR results from different processing approaches
        all_ocr_results = []
        
        # First try with the detected plate region if available
        if plate_region is not None and plate_region.size > 0:
            # Apply different preprocessings and try OCR on each
            preprocessed_images = []
            
            # Original plate region
            preprocessed_images.append(plate_region)
            
            # Grayscale
            gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            preprocessed_images.append(gray_plate)
            
            # CLAHE enhanced
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_plate = clahe.apply(gray_plate)
            preprocessed_images.append(clahe_plate)
            
            # Otsu thresholding
            _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(thresh_plate)
            
            # Adaptive thresholding
            adaptive_plate = cv2.adaptiveThreshold(
                gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            preprocessed_images.append(adaptive_plate)
            
            # Process all preprocessed images with OCR
            for img in preprocessed_images:
                if img is not None and img.size > 0:
                    try:
                        ocr_results = reader.readtext(img, detail=0, paragraph=False)
                        all_ocr_results.extend(ocr_results)
                    except Exception as e:
                        logger.debug(f"OCR error on preprocessed image: {e}")
        
        # If no plate region detected or OCR failed on plate regions, try the whole car
        if not all_ocr_results:
            # Try different preprocessings of the whole car image
            gray_car = cv2.cvtColor(enhanced_car, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold
            adaptive_car = cv2.adaptiveThreshold(
                gray_car, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Try OCR on both
            try:
                ocr_results = reader.readtext(gray_car, detail=0, paragraph=False)
                all_ocr_results.extend(ocr_results)
            except Exception:
                pass
                
            try:
                ocr_results = reader.readtext(adaptive_car, detail=0, paragraph=False)
                all_ocr_results.extend(ocr_results)
            except Exception:
                pass
        
        # Process all OCR results to find the best license plate candidate
        plate_candidates = []
        
        for text in all_ocr_results:
            # Normalize: remove spaces and convert to uppercase
            text = ''.join(c for c in text.upper() if c.isalnum())
            
            # Check if this looks like a license plate
            if len(text) >= 4 and len(text) <= 10 and any(char.isdigit() for char in text):
                plate_candidates.append(text)
        
        # Return the most likely candidate (longest, or most common if tied)
        if plate_candidates:
            # Count occurrences of each candidate
            from collections import Counter
            candidate_counts = Counter(plate_candidates)
            
            # If we have duplicates, pick the most common
            if candidate_counts.most_common(1)[0][1] > 1:
                return candidate_counts.most_common(1)[0][0]
            
            # Otherwise return the longest
            return max(plate_candidates, key=len)
            
        return None
    except Exception as e:
        logger.error(f"Error during LPR: {e}")
        traceback.print_exc()
        return None

def detect_motion(current_frame, previous_frame):
    """Detect motion between frames"""
    if previous_frame is None:
        return False, None
    
    try:
        # Convert frames to grayscale
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
        gray_previous = cv2.GaussianBlur(gray_previous, (21, 21), 0)
        
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
    except Exception as e:
        logger.error(f"Error during motion detection: {e}")
        return False, None

def start_new_recording(cap, frame):
    """Start a new video recording segment"""
    if not ENABLE_CONTINUOUS_RECORDING:
        return None, None
        
    global recording_writer, recording_start_time
    
    try:
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
    except Exception as e:
        logger.error(f"Error starting new recording: {e}")
        return None, None

def save_detection_and_metadata(frame, box, label, full_frame=None, plate_text=None):
    """Save detection image and metadata"""
    try:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Ensure coordinates are within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Skip if box dimensions are invalid
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bounding box dimensions: {x1},{y1},{x2},{y2}")
            return False
            
        detected_object_img = frame[y1:y2, x1:x2]
        
        # Enhance the cropped image
        enhanced_img = enhance_image(detected_object_img)
        if enhanced_img is None:
            logger.warning(f"Image enhancement failed for {label}")
            enhanced_img = detected_object_img
        
        # Check if this is a duplicate image
        if is_duplicate(enhanced_img, label):
            logger.info(f"Skipping duplicate detection: {label}")
            return False
        
        timestamp = datetime.datetime.now()
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{ts_str}.jpg"
        relative_filepath = os.path.join(SAVE_DIR_RELATIVE, filename)
        absolute_filepath = os.path.join(SAVE_DIR_ABSOLUTE, filename)
        
        # Save enhanced cropped object image
        cv2.imwrite(absolute_filepath, enhanced_img)
        
        # If enabled, save the full frame with bounding box
        full_frame_relative = None
        if SAVE_FULL_FRAME and full_frame is not None:
            full_frame_copy = full_frame.copy()
            cv2.rectangle(full_frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label and confidence
            conf_text = f"{label} ({box.conf[0]:.2f})"
            if plate_text:
                conf_text += f" - {plate_text}"
                
            cv2.putText(full_frame_copy, conf_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            full_frame_filename = f"full_{label}_{ts_str}.jpg"
            full_frame_path = os.path.join(SAVE_DIR_ABSOLUTE, full_frame_filename)
            cv2.imwrite(full_frame_path, full_frame_copy)
            
            # Store the relative path to the full frame
            full_frame_relative = os.path.join(SAVE_DIR_RELATIVE, full_frame_filename)
        
        logger.info(f"Saved image: {absolute_filepath}")
        
        # Save metadata to Django database
        detection_record = Detection(
            timestamp=timestamp,
            label=label,
            image_path=relative_filepath,
            confidence=float(box.conf[0]) if box.conf is not None else 0.0
        )
        
        # Add additional fields if your model supports them
        # If you've extended your model, uncomment and use these:
        # detection_record.full_frame_path = full_frame_relative
        # detection_record.license_plate = plate_text
        
        detection_record.save()
        logger.info(f"Saved metadata to DB for: {label}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving detection: {e}")
        traceback.print_exc()
        return False

def generate_daily_report():
    """Generate a summary of today's detections"""
    today = timezone.now().date()
    
    try:
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
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        return None

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
    
    global previous_frame, recording_writer, recording_start_time, running, last_db_cleanup_time, last_motion_check_time
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    frame_count = 0
    processed_count = 0
    skipped_duplicates = 0
    start_time = time.time()
    last_db_cleanup_time = time.time()
    last_motion_check_time = time.time()
    
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
            
            # Run database cleanup if needed
            if DB_CLEANUP_ENABLED and (current_time - last_db_cleanup_time) > (DB_CLEANUP_INTERVAL_HOURS * 3600):
                thread = Thread(target=cleanup_database)
                thread.daemon = True
                thread.start()
                last_db_cleanup_time = current_time
            
            # Perform object detection
            results = perform_object_detection(frame)
            
            if results is not None and results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    detected_label = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Skip low confidence detections
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    
                    # Calculate a position-based ID for this detection
                    cx = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    cy = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    
                    # Make the grid size adaptive to the frame size
                    grid_x = max(50, int(frame.shape[1] / 20))
                    grid_y = max(50, int(frame.shape[0] / 20))
                    
                    pseudo_id = f"{detected_label}_{cx//grid_x}_{cy//grid_y}"
                    
                    # Check if we've recently detected something in this area
                    if current_time - last_detection_time.get(pseudo_id, 0) > DETECTION_INTERVAL:
                        logger.info(f"Detected: {detected_label} (Conf: {confidence:.2f})")
                        
                        # Default label
                        final_label = detected_label
                        plate_text = None
                        
                        # Process cars differently to detect license plates
                        if detected_label == 'car' or detected_label == 'truck':
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Ensure coordinates are within frame boundaries
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            
                            # Skip if box dimensions are invalid
                            if x2 <= x1 or y2 <= y1:
                                continue
                                
                            car_image = frame[y1:y2, x1:x2]
                            plate_text = perform_lpr(car_image)
                            
                            if plate_text:
                                logger.info(f"  LPR Result: {plate_text}")
                                final_label = f"car_plate_{plate_text}"
                            else:
                                final_label = "car_no_plate"
                        
                        # Save the detection
                        saved = save_detection_and_metadata(frame, box, final_label, full_frame=frame, plate_text=plate_text)
                        
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
                
                # Reset counters periodically to get more recent stats
                if frame_count > 10000:
                    frame_count = 0
                    processed_count = 0
                    skipped_duplicates = 0
                    start_time = time.time()
            
            # Generate daily report at midnight
            current_hour = datetime.datetime.now().hour
            current_minute = datetime.datetime.now().minute
            if current_hour == 0 and current_minute == 0 and frame_count % 100 == 0:
                thread = Thread(target=generate_daily_report)
                thread.daemon = True
                thread.start()
                
                thread = Thread(target=cleanup_old_recordings)
                thread.daemon = True
                thread.start()
            
    except KeyboardInterrupt:
        logger.info("Stream capture stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        logger.info("Releasing video capture resource.")
        
        # Close video writer if open
        if recording_writer is not None:
            recording_writer.release()
        
        cap.release()
        
        # Generate final report
        generate_daily_report()

if __name__ == "__main__":
    connect_capture_detect(RTSP_URL)