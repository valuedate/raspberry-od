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
MAX_CACHE_IMAGES = 100

# --- Duplicate Detection Settings ---
SIMILARITY_THRESHOLD = 0.80  # Higher values require more similarity to consider as duplicate
HASH_SIZE = 32  # Size of the perceptual hash (higher = more detail)
DETECTION_INTERVAL = 3  # Seconds between detections at the same location
MIN_DETECTION_INTERVAL = 0.5  # Minimum interval between any detections

# --- Image Quality Improvement ---
USE_SUPER_RESOLUTION = False  # Enable if you have GPU for real-time processing
ENHANCE_CONTRAST = True
DENOISE_IMAGES = True
SHARPEN_IMAGES = True

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
    """Create a more robust perceptual hash of the image for duplicate detection"""
    # Calculate structural similarity-based hash
    # First ensure the image is not empty
    if image.size == 0 or image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return 0
        
    # Resize and convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (HASH_SIZE, HASH_SIZE))
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Calculate gradient in both directions to capture edges
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    
    # Calculate gradient magnitude
    mag = cv2.magnitude(gx, gy)
    
    # Normalize
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Binarize
    _, mag_bin = cv2.threshold(mag, 127, 255, cv2.THRESH_BINARY)
    
    # Create hash
    hash_value = 0
    for i in range(HASH_SIZE):
        for j in range(HASH_SIZE):
            if mag_bin[i, j] > 0:
                hash_value |= 1 << (i * HASH_SIZE + j)
    
    return hash_value

def hamming_distance(hash1, hash2):
    """Calculate the Hamming distance between two hashes"""
    return bin(hash1 ^ hash2).count('1')

def is_duplicate(image, label_type):
    """Check if an image is a duplicate using multiple methods for robustness"""
    # Skip empty or tiny images
    if image is None or image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
        return False
    
    img_hash = image_hash(image)
    
    if label_type not in recent_image_hashes:
        recent_image_hashes[label_type] = []
    
    # Calculate image features for more robust matching
    # 1. Calculate histogram
    hist_img = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_img, hist_img, 0, 1, cv2.NORM_MINMAX)
    hist_img = hist_img.flatten()
    
    current_time = time.time()
    matches = 0
    total_comparisons = 0
    
    for (stored_hash, stored_time, stored_hist) in recent_image_hashes[label_type]:
        total_comparisons += 1
        
        # Skip very old entries
        if current_time - stored_time > 3600:  # 1 hour
            continue
        
        # Step 1: Quick hash-based comparison
        distance = hamming_distance(img_hash, stored_hash)
        max_distance = HASH_SIZE * HASH_SIZE
        hash_similarity = 1 - (distance / max_distance)
        
        # If hash similarity is low, skip to next comparison
        if hash_similarity < SIMILARITY_THRESHOLD - 0.2:
            continue
            
        # Step 2: More detailed histogram comparison
        hist_correlation = cv2.compareHist(hist_img, stored_hist, cv2.HISTCMP_CORREL)
        
        # Combined similarity metric
        combined_similarity = (hash_similarity * 0.6) + (hist_correlation * 0.4)
        
        if combined_similarity >= SIMILARITY_THRESHOLD:
            logger.debug(f"Duplicate found: hash_sim={hash_similarity:.2f}, hist_corr={hist_correlation:.2f}, combined={combined_similarity:.2f}")
            matches += 1
            
            # If we've found multiple matches, it's very likely a duplicate
            if matches >= 2:
                return True
    
    # Add this hash and histogram to the recent entries
    hist_img_entry = hist_img.copy()
    recent_image_hashes[label_type].append((img_hash, current_time, hist_img_entry))
    
    # Periodically clean up old entries
    if len(recent_image_hashes[label_type]) > MAX_CACHE_IMAGES:
        # Sort by time (newest first)
        recent_image_hashes[label_type].sort(key=lambda x: x[1], reverse=True)
        # Keep only most recent entries
        recent_image_hashes[label_type] = recent_image_hashes[label_type][:MAX_CACHE_IMAGES]
    
    # If we did many comparisons but found only one similar image, 
    # it might be a false positive, so we allow it
    return matches > 0 and (total_comparisons < 5 or matches > 1)

def perform_object_detection(frame):
    results = model(frame, classes=TARGET_CLASSES, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
    return results[0]

def enhance_image(image):
    """Apply various image enhancement techniques"""
    if image is None or image.size == 0:
        return image
        
    # Make a copy to avoid modifying the original
    enhanced = image.copy()
    
    # 1. Convert to LAB color space and apply CLAHE to L channel
    if ENHANCE_CONTRAST:
        if len(enhanced.shape) == 3:  # Color image
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
    
    # 2. Denoise the image
    if DENOISE_IMAGES:
        if len(enhanced.shape) == 3:  # Color image
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        else:  # Grayscale
            enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # 3. Sharpen the image
    if SHARPEN_IMAGES:
        kernel = np.array([[-1, -1, -1], 
                          [-1, 9, -1], 
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # 4. Apply super-resolution if enabled and available
    if USE_SUPER_RESOLUTION and device == 'cuda':
        try:
            # This requires OpenCV with DNN Super Resolution module
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            path = "models/ESPCN_x2.pb"  # Assumes you have this model file
            if os.path.exists(path):
                sr.readModel(path)
                sr.setModel("espcn", 2)  # 2x upscaling
                enhanced = sr.upsample(enhanced)
                # Resize back down to a reasonable size if it became too large
                if enhanced.shape[0] > 800 or enhanced.shape[1] > 800:
                    scale = min(800 / enhanced.shape[0], 800 / enhanced.shape[1])
                    enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale)
        except Exception as e:
            logger.warning(f"Super-resolution failed: {e}")
    
    return enhanced

def detect_license_plate(car_image):
    """Attempt to locate the license plate within the car image"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges
        edged = cv2.Canny(gray, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Loop over contours to find rectangle-like shapes
        plate_contour = None
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # License plates are roughly rectangular with 4 corners
            if len(approx) == 4:
                # Check aspect ratio to filter out non-plate rectangles
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # Most license plates have aspect ratio between 1.5 and 5
                if 1.5 <= aspect_ratio <= 5.0:
                    # Check minimum size
                    if w > 80 and h > 20:
                        plate_contour = approx
                        break
        
        if plate_contour is not None:
            # Get ROI
            x, y, w, h = cv2.boundingRect(plate_contour)
            
            # Expand the region slightly
            padding_x = int(w * 0.05)
            padding_y = int(h * 0.1)
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(car_image.shape[1] - x, w + 2 * padding_x)
            h = min(car_image.shape[0] - y, h + 2 * padding_y)
            
            plate_img = car_image[y:y+h, x:x+w]
            
            # Return the potential plate region
            if plate_img.size > 0:
                return plate_img, True
    
    except Exception as e:
        logger.warning(f"Error detecting license plate: {e}")
    
    # If no plate found or error, return the original image
    return car_image, False

def perform_lpr(car_image):
    """Enhanced license plate recognition with multiple image processing techniques"""
    try:
        # Skip processing if image is empty
        if car_image is None or car_image.size == 0:
            return None
        
        # Store original image
        original_car_image = car_image.copy()
        
        # First try to detect the license plate region
        plate_image, plate_found = detect_license_plate(car_image)
        
        # Enhance the image quality
        enhanced_plate = enhance_image(plate_image)
        
        # Create multiple versions of the image to try OCR on
        images_to_try = []
        
        # 1. Enhanced original
        images_to_try.append(enhanced_plate)
        
        # 2. Grayscale with adaptive threshold
        gray_plate = cv2.cvtColor(enhanced_plate, cv2.COLOR_BGR2GRAY)
        binary_adaptive = cv2.adaptiveThreshold(
            gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        images_to_try.append(binary_adaptive)
        
        # 3. Grayscale with Otsu's threshold
        _, binary_otsu = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images_to_try.append(binary_otsu)
        
        # 4. Edge-enhanced version
        edge_enhanced = cv2.Canny(gray_plate, 100, 200)
        # Dilate edges to make them more visible
        kernel = np.ones((2, 2), np.uint8)
        edge_enhanced = cv2.dilate(edge_enhanced, kernel, iterations=1)
        images_to_try.append(edge_enhanced)
        
        # 5. Morphological transformations
        kernel = np.ones((3, 3), np.uint8)
        morph_img = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel)
        morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel)
        images_to_try.append(morph_img)
        
        # If we couldn't find a plate region, also try with the original car image
        if not plate_found:
            # Convert full car to grayscale with adaptive threshold
            gray_car = cv2.cvtColor(enhance_image(original_car_image), cv2.COLOR_BGR2GRAY)
            binary_adaptive_car = cv2.adaptiveThreshold(
                gray_car, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            images_to_try.append(binary_adaptive_car)
        
        # Try OCR on all image variants
        all_results = []
        
        for img in images_to_try:
            # Set OCR parameters based on what we're looking for
            ocr_results = reader.readtext(
                img, 
                detail=0,
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                batch_size=1,
                beamWidth=5,
                contrast_ths=0.3,
                adjust_contrast=0.5
            )
            
            for text in ocr_results:
                # Clean and validate the text
                text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
                
                # Check if it's likely a license plate
                if (len(text) >= 4 and 
                    len(text) <= 10 and  # Most plates are 4-10 characters
                    text.isalnum() and 
                    any(char.isdigit() for char in text) and
                    any(char.isalpha() for char in text)):
                    all_results.append(text)
        
        # If we found potential plates, choose the most likely one
        if all_results:
            # Count occurrences of each result
            from collections import Counter
            result_counts = Counter(all_results)
            
            # If any result appears multiple times, it's more likely correct
            if result_counts.most_common(1)[0][1] > 1:
                return result_counts.most_common(1)[0][0]
            
            # Otherwise return the longest result
            return max(all_results, key=len)
            
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

def enhance_and_save_image(image, filepath):
    """Enhance image quality before saving to disk"""
    # Apply enhancement
    enhanced = enhance_image(image)
    
    # Apply compression parameters to maintain quality
    params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # 95% quality JPEG
    
    # Save with enhanced quality
    cv2.imwrite(filepath, enhanced, params)
    return enhanced

def save_detection_and_metadata(frame, box, label, full_frame=None):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Ensure box coordinates are within frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    # Skip if the box is too small
    if x2 - x1 < 10 or y2 - y1 < 10:
        logger.warning(f"Skipping detection with too small bounding box: {label}")
        return False
    
    detected_object_img = frame[y1:y2, x1:x2]
    
    # Skip if image extraction failed
    if detected_object_img is None or detected_object_img.size == 0:
        logger.warning(f"Skipping detection with empty image: {label}")
        return False
    
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
        # Enhance and save the cropped image
        enhanced_obj_img = enhance_and_save_image(detected_object_img, absolute_filepath)
        
        # If enabled, save the full frame with bounding box
        if SAVE_FULL_FRAME and full_frame is not None:
            full_frame_copy = full_frame.copy()
            
            # Draw a better looking bounding box
            cv2.rectangle(full_frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add a semi-transparent background for the text
            text_background = full_frame_copy.copy()
            cv2.rectangle(text_background, (x1, y1 - 30), (x1 + len(label) * 12, y1), (0, 0, 0), -1)
            cv2.addWeighted(text_background, 0.6, full_frame_copy, 0.4, 0, full_frame_copy)
            
            # Add text with better visibility
            cv2.putText(full_frame_copy, label, (x1 + 5, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            full_frame_filename = f"full_{label}_{ts_str}.jpg"
            full_frame_path = os.path.join(SAVE_DIR_ABSOLUTE, full_frame_filename)
            
            # Enhance and save full frame
            enhance_and_save_image(full_frame_copy, full_frame_path)
            
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
        )
        
        # If you extend your model to include full frame path, uncomment:
        # if full_frame_relative:
        #     detection_record.full_frame_path = full_frame_relative
        
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

_time is None:
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