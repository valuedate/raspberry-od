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
import queue
from dataclasses import dataclass, field
import uuid

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
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
# Create a dataclass to hold camera-specific configurations
@dataclass
class CameraConfig:
    camera_id: str
    username: str
    password: str
    ip: str
    channel: int = 1
    stream_type: int = 1  # 1 for main stream, 2 for sub stream
    rtsp_url: str = ""

    def __post_init__(self):
        # Construct the RTSP URL after initialization
        self.rtsp_url = f"rtsp://{self.username}:{self.password}@{self.ip}:554/Streaming/channels/{self.channel}0{self.stream_type}"


# Define multiple camera configurations here
CAMERA_CONFIGS = [
    CameraConfig(
        camera_id="camera_115",
        username="admin",
        password="rFERNANDES18",
        ip="10.0.0.115"
    ),
    CameraConfig(
        camera_id="camera_111",
        username="admin",
        password="rFERNANDES18",
        ip="10.0.0.111"
    ),
    CameraConfig(
        camera_id="camera_112",
        username="admin",
        password="rFERNANDES18",
        ip="10.0.0.112"
    ),
    CameraConfig(
        camera_id="camera_113",
        username="admin",
        password="rFERNANDES18",
        ip="10.0.0.113"
    ),
]

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
MAX_CACHE_IMAGES = 250

# --- Duplicate Detection Settings ---
# Replaced with a more robust tracking system
# SIMILARITY_THRESHOLD = 0.90
HASH_SIZE = 32
DETECTION_INTERVAL = 5  # Seconds between detections at the same location
MIN_DETECTION_INTERVAL = 1  # Minimum interval between any detections
STALL_DETECTION_TIME = 10  # Seconds an object must be stationary to be considered stalled
MAX_TRACKING_DISTANCE = 100  # Maximum pixel distance to match a detection to a track
TRACK_LOSS_THRESHOLD = 15  # Seconds before a track is considered lost
MIN_MOVEMENT_THRESHOLD = 10  # Minimum pixel movement to not be considered stalled

# --- Advanced Settings ---
SAVE_FULL_FRAME = True
ENABLE_CONTINUOUS_RECORDING = False
CONTINUOUS_RECORDING_DIR = os.path.join(PROJECT_DIR, "media", "recordings")
RECORDING_SEGMENT_MINUTES = 10
MAX_RECORDING_DAYS = 7

# --- Motion Detection ---
ENABLE_MOTION_DETECTION = True
MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 1000
MOTION_DETECTION_INTERVAL = 0.5

# --- License Plate Recognition Settings ---
LPR_MIN_PLATE_ASPECT_RATIO = 1.5
LPR_MAX_PLATE_ASPECT_RATIO = 5.0
LPR_MIN_PLATE_AREA_RATIO = 0.005
LPR_MAX_PLATE_AREA_RATIO = 0.15

# --- Database Cleanup Settings ---
DB_CLEANUP_ENABLED = True
DB_CLEANUP_DAYS = 30
DB_CLEANUP_INTERVAL_HOURS = 24

# --- Initialization ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

logger.info(f"Loading YOLO model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
model.to(device)

logger.info("Loading EasyOCR reader...")
reader = easyocr.Reader(["en"], gpu=(device == "cuda"))

os.makedirs(SAVE_DIR_ABSOLUTE, exist_ok=True)
logger.info(f"Saving detections to: {SAVE_DIR_ABSOLUTE}")

if ENABLE_CONTINUOUS_RECORDING:
    os.makedirs(CONTINUOUS_RECORDING_DIR, exist_ok=True)
    logger.info(f"Continuous recording enabled. Saving to: {CONTINUOUS_RECORDING_DIR}")

# --- Global State Variables for Cleanup and Reports ---
cleanup_queue = queue.Queue()
running = True


# --- New Dataclasses for Tracking ---
@dataclass
class TrackedObject:
    """Represents a single tracked object across frames."""
    track_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    centroid: tuple[int, int] = (0, 0)
    last_seen: float = 0.0
    last_saved: float = 0.0
    is_stalled: bool = False
    initial_save_done: bool = False
    db_record_id: int = None
    initial_centroid: tuple[int, int] = (0, 0)


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


def enhance_image(image):
    """Apply image enhancement techniques"""
    if image is None or image.size == 0:
        return None

    try:
        height, width = image.shape[:2]
        scale = min(2.0, max(1.0, 800 / max(width, height)))
        if scale > 1.0:
            image = cv2.resize(image, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)

        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return image


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
        gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(filtered, 30, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        car_area = car_image.shape[0] * car_image.shape[1]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            if len(approx) >= 4 and len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                area_ratio = (w * h) / car_area

                if (LPR_MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= LPR_MAX_PLATE_ASPECT_RATIO and
                        LPR_MIN_PLATE_AREA_RATIO <= area_ratio <= LPR_MAX_PLATE_AREA_RATIO):
                    margin_x = int(w * 0.05)
                    margin_y = int(h * 0.1)

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
        enhanced_car = enhance_image(car_image)
        plate_region = detect_license_plate_region(enhanced_car)
        all_ocr_results = []

        if plate_region is not None and plate_region.size > 0:
            preprocessed_images = [
                plate_region,
                cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            ]

            gray_plate = preprocessed_images[1]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            preprocessed_images.append(clahe.apply(gray_plate))
            _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(thresh_plate)

            adaptive_plate = cv2.adaptiveThreshold(
                gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            preprocessed_images.append(adaptive_plate)

            for img in preprocessed_images:
                if img is not None and img.size > 0:
                    try:
                        ocr_results = reader.readtext(img, detail=0, paragraph=False)
                        all_ocr_results.extend(ocr_results)
                    except Exception as e:
                        logger.debug(f"OCR error on preprocessed image: {e}")

        if not all_ocr_results:
            gray_car = cv2.cvtColor(enhanced_car, cv2.COLOR_BGR2GRAY)
            adaptive_car = cv2.adaptiveThreshold(
                gray_car, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            try:
                all_ocr_results.extend(reader.readtext(gray_car, detail=0, paragraph=False))
            except Exception:
                pass

            try:
                all_ocr_results.extend(reader.readtext(adaptive_car, detail=0, paragraph=False))
            except Exception:
                pass

        plate_candidates = []
        for text in all_ocr_results:
            text = ''.join(c for c in text.upper() if c.isalnum())
            if len(text) >= 4 and len(text) <= 10 and any(char.isdigit() for char in text):
                plate_candidates.append(text)

        if plate_candidates:
            from collections import Counter
            candidate_counts = Counter(plate_candidates)
            if candidate_counts.most_common(1)[0][1] > 1:
                return candidate_counts.most_common(1)[0][0]
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
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
        gray_previous = cv2.GaussianBlur(gray_previous, (21, 21), 0)
        frame_diff = cv2.absdiff(gray_current, gray_previous)
        thresh = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > MOTION_MIN_AREA]

        motion_overlay = current_frame.copy()
        cv2.drawContours(motion_overlay, significant_contours, -1, (0, 255, 0), 2)

        return len(significant_contours) > 0, motion_overlay
    except Exception as e:
        logger.error(f"Error during motion detection: {e}")
        return False, None


def start_new_recording(cap, frame, camera_id):
    """Start a new video recording segment"""
    if not ENABLE_CONTINUOUS_RECORDING:
        return None, None

    try:
        timestamp = datetime.datetime.now()
        filename = f"{camera_id}_recording_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = os.path.join(CONTINUOUS_RECORDING_DIR, filename)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25  # Fallback to 25 FPS

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        recording_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        recording_start_time = timestamp

        recording_writer.write(frame)

        logger.info(f"Started new recording segment for {camera_id}: {filename}")
        return recording_writer, recording_start_time
    except Exception as e:
        logger.error(f"Error starting new recording for {camera_id}: {e}")
        return None, None


def save_detection_and_metadata(frame, box, label, camera_id, position_id, full_frame=None, plate_text=None,
                                db_record_id=None, update_existing=False):
    """Save detection image and metadata. Updated to handle existing records."""
    try:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bounding box dimensions: {x1},{y1},{x2},{y2}")
            return None

        detected_object_img = frame[y1:y2, x1:x2]
        enhanced_img = enhance_image(detected_object_img)
        if enhanced_img is None:
            enhanced_img = detected_object_img

        timestamp = datetime.datetime.now()
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{camera_id}_{label}_{ts_str}.jpg"
        relative_filepath = os.path.join(SAVE_DIR_RELATIVE, filename)
        absolute_filepath = os.path.join(SAVE_DIR_ABSOLUTE, filename)

        cv2.imwrite(absolute_filepath, enhanced_img)

        # Handle full frame with annotation
        full_frame_relative = None
        if SAVE_FULL_FRAME and full_frame is not None:
            full_frame_copy = full_frame.copy()
            cv2.rectangle(full_frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf_text = f"{label} ({box.conf[0]:.2f})"
            if plate_text:
                conf_text += f" - {plate_text}"
            cv2.putText(full_frame_copy, conf_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            full_frame_filename = f"full_{camera_id}_{label}_{ts_str}.jpg"
            full_frame_path = os.path.join(SAVE_DIR_ABSOLUTE, full_frame_filename)
            cv2.imwrite(full_frame_path, full_frame_copy)
            full_frame_relative = os.path.join(SAVE_DIR_RELATIVE, full_frame_filename)

        logger.info(f"Saved image for {camera_id}: {absolute_filepath}")

        if update_existing and db_record_id:
            detection_record = Detection.objects.get(pk=db_record_id)
            detection_record.last_updated = timestamp
            detection_record.image_path = relative_filepath
            # Optional: update other fields like confidence if needed
            detection_record.save()
            logger.info(f"Updated existing DB record {db_record_id} for {camera_id}: {label}")
            return db_record_id
        else:
            detection_record = Detection(
                timestamp=timestamp,
                label=label,
                image_path=relative_filepath,
                camera=camera_id,
                position=position_id,
                confidence=float(box.conf[0])
            )
            detection_record.save()
            logger.info(f"Saved NEW metadata to DB for {camera_id}: {label} at position {position_id}")
            return detection_record.pk
    except Exception as e:
        logger.error(f"Error saving detection for {camera_id}: {e}")
        traceback.print_exc()
        return None


def generate_daily_report():
    """Generate a summary of today's detections"""
    today = timezone.now().date()

    try:
        all_stats = Detection.objects.filter(
            timestamp__date=today
        ).values('camera', 'label').annotate(count=Count('label')).order_by('camera', '-count')

        report = f"Daily Detection Report for {today}\n"
        report += "=" * 40 + "\n\n"

        current_camera = None
        for stat in all_stats:
            if stat['camera'] != current_camera:
                current_camera = stat['camera']
                report += f"\nCamera ID: {current_camera}\n"
                report += "-" * 20 + "\n"
            report += f"- {stat['label']}: {stat['count']}\n"

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
    logger.info("Termination signal received. Shutting down all cameras...")
    running = False


def run_cleanup_tasks():
    """A separate thread to handle cleanup and report generation"""
    while running:
        try:
            task = cleanup_queue.get(timeout=60)
            if task == "db_cleanup":
                cleanup_database()
            elif task == "daily_report":
                generate_daily_report()
            elif task == "recording_cleanup":
                cleanup_old_recordings()
            cleanup_queue.task_done()
        except queue.Empty:
            # Check for a cleanup task once per day at midnight
            current_time = datetime.datetime.now()
            if current_time.hour == 0 and current_time.minute == 1:
                cleanup_queue.put("db_cleanup")
                cleanup_queue.put("daily_report")
                cleanup_queue.put("recording_cleanup")
            time.sleep(60)  # Sleep for a minute
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}")
            time.sleep(60)


# Dictionary to hold per-camera state
per_camera_state = {}


def connect_capture_detect(camera_config: CameraConfig):
    """The main loop for a single camera feed."""
    logger.info(
        f"[{camera_config.camera_id}] Attempting to connect to: {camera_config.rtsp_url.replace(camera_config.password, '********')}")
    cap = cv2.VideoCapture(camera_config.rtsp_url)

    if not cap.isOpened():
        logger.error(f"[{camera_config.camera_id}] Error: Could not open video stream.")
        return

    logger.info(f"[{camera_config.camera_id}] Successfully connected to the camera stream.")

    # Initialize per-camera state
    per_camera_state[camera_config.camera_id] = {
        'previous_frame': None,
        'recording_writer': None,
        'recording_start_time': None,
        'last_motion_check_time': 0,
        'active_tracks': {},
    }

    state = per_camera_state[camera_config.camera_id]

    frame_count = 0
    start_time = time.time()

    try:
        while running:
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"[{camera_config.camera_id}] Failed to retrieve frame. Reconnecting...")
                time.sleep(5)
                cap.release()
                cap = cv2.VideoCapture(camera_config.rtsp_url)
                if not cap.isOpened():
                    logger.error(f"[{camera_config.camera_id}] Failed to reconnect. Exiting thread.")
                    break
                else:
                    logger.info(f"[{camera_config.camera_id}] Reconnected successfully.")
                    continue

            current_time = time.time()
            frame_count += 1

            # Continuous recording
            if ENABLE_CONTINUOUS_RECORDING:
                if state['recording_writer'] is None:
                    state['recording_writer'], state['recording_start_time'] = start_new_recording(cap, frame,
                                                                                                   camera_config.camera_id)
                else:
                    segment_duration = datetime.timedelta(minutes=RECORDING_SEGMENT_MINUTES)
                    current_datetime = datetime.datetime.now()
                    if current_datetime - state['recording_start_time'] > segment_duration:
                        state['recording_writer'], state['recording_start_time'] = start_new_recording(cap, frame,
                                                                                                       camera_config.camera_id)
                    else:
                        state['recording_writer'].write(frame)

            # Motion detection
            motion_detected = False
            if ENABLE_MOTION_DETECTION and current_time - state['last_motion_check_time'] >= MOTION_DETECTION_INTERVAL:
                motion_detected, _ = detect_motion(frame, state['previous_frame'])
                state['last_motion_check_time'] = current_time
                state['previous_frame'] = frame.copy()

                if not motion_detected and frame_count % 30 != 0:
                    continue

            if frame_count % 3 != 0 and not motion_detected:
                continue

            results = perform_object_detection(frame)
            new_detections = []
            if results and results.boxes:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    detected_label = model.names[class_id]
                    confidence = float(box.conf[0])
                    if confidence >= CONFIDENCE_THRESHOLD:
                        new_detections.append({'box': box, 'label': detected_label,
                                               'centroid': ((box.xyxy[0][0] + box.xyxy[0][2]) / 2,
                                                            (box.xyxy[0][1] + box.xyxy[0][3]) / 2)})

            unmatched_detections = new_detections.copy()

            # Update existing tracks and check for stalled objects
            for track_id, track in list(state['active_tracks'].items()):
                best_match_idx = -1
                min_distance = float('inf')

                # Find the best match for the existing track
                for i, new_det in enumerate(unmatched_detections):
                    distance = np.linalg.norm(np.array(track.centroid) - np.array(new_det['centroid']))
                    if distance < min_distance and distance < MAX_TRACKING_DISTANCE:
                        min_distance = distance
                        best_match_idx = i

                if best_match_idx != -1:
                    # Match found! Update the track with new information
                    matched_det = unmatched_detections.pop(best_match_idx)

                    # Check for stalling
                    movement = np.linalg.norm(np.array(track.initial_centroid) - np.array(matched_det['centroid']))

                    if movement < MIN_MOVEMENT_THRESHOLD:
                        if current_time - track.last_seen > STALL_DETECTION_TIME:
                            track.is_stalled = True
                    else:
                        track.is_stalled = False
                        track.initial_centroid = matched_det['centroid']  # Reset initial centroid when moving

                    # Update track state
                    track.centroid = matched_det['centroid']
                    track.last_seen = current_time

                    # Save detection logic
                    if track.initial_save_done:
                        if track.is_stalled and current_time - track.last_saved > DETECTION_INTERVAL:
                            logger.info(
                                f"[{camera_config.camera_id}] Stalled object ({track.label}) detected. Updating record...")
                            db_record_id = save_detection_and_metadata(
                                frame, matched_det['box'], track.label, camera_config.camera_id,
                                str(track.track_id), full_frame=frame, update_existing=True,
                                db_record_id=track.db_record_id
                            )
                            if db_record_id:
                                track.last_saved = current_time
                        elif not track.is_stalled:
                            logger.info(
                                f"[{camera_config.camera_id}] Moving object ({track.label}) detected. Saving new record...")
                            db_record_id = save_detection_and_metadata(
                                frame, matched_det['box'], track.label, camera_config.camera_id,
                                str(track.track_id), full_frame=frame
                            )
                            if db_record_id:
                                track.db_record_id = db_record_id
                                track.last_saved = current_time

                    else:  # Initial save for this track
                        logger.info(
                            f"[{camera_config.camera_id}] New object ({track.label}) detected. Saving initial record...")
                        db_record_id = save_detection_and_metadata(
                            frame, matched_det['box'], track.label, camera_config.camera_id,
                            str(track.track_id), full_frame=frame
                        )
                        if db_record_id:
                            track.db_record_id = db_record_id
                            track.initial_save_done = True
                            track.last_saved = current_time

                else:
                    # No match found, track might be lost
                    if current_time - track.last_seen > TRACK_LOSS_THRESHOLD:
                        del state['active_tracks'][track_id]
                        logger.info(f"[{camera_config.camera_id}] Track {track_id} for {track.label} lost and removed.")

            # Create new tracks for unmatched detections
            for new_det in unmatched_detections:
                new_track = TrackedObject(
                    label=new_det['label'],
                    centroid=new_det['centroid'],
                    last_seen=current_time,
                    last_saved=current_time,
                    initial_centroid=new_det['centroid']
                )
                state['active_tracks'][new_track.track_id] = new_track

                # Immediately save the first detection for this new track
                logger.info(
                    f"[{camera_config.camera_id}] New object ({new_track.label}) detected. Saving initial record...")
                db_record_id = save_detection_and_metadata(
                    frame, new_det['box'], new_track.label, camera_config.camera_id,
                    str(new_track.track_id), full_frame=frame
                )
                if db_record_id:
                    new_track.db_record_id = db_record_id
                    new_track.initial_save_done = True

            # Check for a single cleanup trigger time (e.g., midnight)
            if datetime.datetime.now().hour == 0 and datetime.datetime.now().minute == 1 and frame_count % 100 == 0:
                cleanup_queue.put("db_cleanup")
                cleanup_queue.put("daily_report")
                cleanup_queue.put("recording_cleanup")

    except Exception as e:
        logger.error(f"[{camera_config.camera_id}] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        logger.info(f"[{camera_config.camera_id}] Releasing video capture resource.")
        if state['recording_writer'] is not None:
            state['recording_writer'].release()
        cap.release()


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the cleanup and reporting thread
    cleanup_thread = Thread(target=run_cleanup_tasks, name="CleanupThread")
    cleanup_thread.daemon = True
    cleanup_thread.start()

    camera_threads = []

    for config in CAMERA_CONFIGS:
        thread_name = f"CameraThread-{config.camera_id}"
        thread = Thread(target=connect_capture_detect, args=(config,), name=thread_name)
        thread.daemon = True
        camera_threads.append(thread)
        thread.start()

    try:
        while running:
            # Main thread sleeps while the others work
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Main thread received KeyboardInterrupt. Triggering graceful shutdown.")
        running = False
    finally:
        for thread in camera_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        if cleanup_thread.is_alive():
            cleanup_thread.join(timeout=5)

        logger.info("All threads have been shut down. Exiting.")