from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseServerError
from django.core.paginator import Paginator
from django.db.models import Q
from .models import Detection
import cv2
import time
import threading

# --- Configuration (Should match detector_django.py or be centralized) ---
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "123456"
CAMERA_IP = "10.0.0.115"
CHANNEL = 1
STREAM_TYPE = 2  # Use sub-stream (lower resolution) for web streaming efficiency
RTSP_URL = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMERA_IP}:554/Streaming/channels/{CHANNEL}0{STREAM_TYPE}"

# --- Global variable for streaming frame ---
output_frame = None
lock = threading.Lock()
stream_thread = None
stop_stream_event = threading.Event()

def capture_frames():
    """Thread function to continuously capture frames from the camera."""
    global output_frame, lock
    print(f"Starting frame capture thread for URL: {RTSP_URL.replace(CAMERA_PASSWORD, '********')}")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream for live view.")
        return

    print("Successfully connected to camera for live view.")
    while not stop_stream_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame for live view. Retrying...")
            cap.release()
            time.sleep(5) # Wait before retrying connection
            cap = cv2.VideoCapture(RTSP_URL)
            if not cap.isOpened():
                print("Error: Failed to reconnect for live view.")
                time.sleep(10) # Wait longer before next attempt
                continue
            else:
                print("Reconnected successfully for live view.")
                continue

        with lock:
            output_frame = frame.copy()
        # Reduce capture rate slightly to lower CPU usage if needed
        time.sleep(0.03) # Approx 30fps target

    print("Stopping frame capture thread.")
    cap.release()

def generate_stream():
    """Generator function to yield JPEG frames for streaming response."""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                # Optionally, return a placeholder image or wait
                # For now, just skip if no frame is available yet
                time.sleep(0.1)
                continue
            # Encode frame as JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            # Ensure the frame was successfully encoded
            if not flag:
                continue

        # Yield the frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        # Control frame rate for streaming
        time.sleep(0.03) # Match capture rate or slightly slower

def live_stream_view(request):
    """View function for the live stream page."""
    global stream_thread
    # Start the background frame capture thread if it's not running
    if stream_thread is None or not stream_thread.is_alive():
        print("Starting background frame capture thread...")
        stop_stream_event.clear()
        stream_thread = threading.Thread(target=capture_frames)
        stream_thread.daemon = True
        stream_thread.start()

    return render(request, 'viewer/live.html')

def video_feed(request):
    """Returns the streaming response."""
    # Check if capture thread is running and connected
    # (Basic check, more robust checks could be added)
    time.sleep(1) # Give thread time to connect initially
    with lock:
        if output_frame is None and (stream_thread is None or not stream_thread.is_alive()):
             return HttpResponseServerError("Camera stream not available.")

    return StreamingHttpResponse(generate_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def history_view(request):
    """View function for the detection history page with search and pagination."""
    query = request.GET.get('q', '')
    detection_list = Detection.objects.all() # Ordered by timestamp descending (defined in model Meta)

    if query:
        # Search in label (covers person, car, plate)
        detection_list = detection_list.filter(Q(label__icontains=query))

    paginator = Paginator(detection_list, 15) # Show 15 detections per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'query': query,
    }
    return render(request, 'viewer/history.html', context)

# Optional: Add a view to stop the stream thread if needed, e.g., when navigating away
# This requires more complex handling (e.g., JavaScript on page unload)
# For simplicity, the thread runs as long as the Django server runs.

