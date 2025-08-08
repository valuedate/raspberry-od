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


from django.views.generic import TemplateView
from django.utils import timezone
from django.db.models import Count, Q
import json
from datetime import timedelta

from .models import Detection

class DashboardView(TemplateView):
    template_name = 'viewer/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get current date info
        today = timezone.now().date()
        yesterday = today - timedelta(days=1)
        
        # Calculate all-time statistics
        context['total_detections'] = Detection.objects.count()
        context['vehicle_count'] = Detection.objects.filter(
            Q(label='car_no_plate') | Q(label__startswith='car_plate_')
        ).count()
        context['vehicle_no_plate_count'] = Detection.objects.filter(label='car_no_plate').count()
        context['person_count'] = Detection.objects.filter(label='person').count()
        context['plate_count'] = Detection.objects.filter(label__startswith='car_plate_').count()
        context['other_count'] = context['total_detections'] - context['vehicle_count'] - context['person_count']
        
        # Calculate today's statistics
        context['today_total'] = Detection.objects.filter(timestamp__date=today).count()
        context['today_vehicles'] = Detection.objects.filter(
            (Q(label='car_no_plate') | Q(label__startswith='car_plate_')) & 
            Q(timestamp__date=today)
        ).count()
        context['today_people'] = Detection.objects.filter(
            label='person', 
            timestamp__date=today
        ).count()
        context['today_plates'] = Detection.objects.filter(
            label__startswith='car_plate_', 
            timestamp__date=today
        ).count()
        
        # Calculate yesterday's statistics for variation
        yesterday_total = Detection.objects.filter(timestamp__date=yesterday).count()
        yesterday_vehicles = Detection.objects.filter(
            (Q(label='car_no_plate') | Q(label__startswith='car_plate_')) & 
            Q(timestamp__date=yesterday)
        ).count()
        yesterday_people = Detection.objects.filter(
            label='person', 
            timestamp__date=yesterday
        ).count()
        yesterday_plates = Detection.objects.filter(
            label__startswith='car_plate_', 
            timestamp__date=yesterday
        ).count()
        
        # Calculate percentage variations (avoid division by zero)
        context['total_variation'] = self._calculate_variation(
            context['today_total'], yesterday_total)
        context['vehicle_variation'] = self._calculate_variation(
            context['today_vehicles'], yesterday_vehicles)
        context['person_variation'] = self._calculate_variation(
            context['today_people'], yesterday_people)
        context['plate_variation'] = self._calculate_variation(
            context['today_plates'], yesterday_plates)
        
        # Get data for the last 7 days
        last_7_days = []
        last_7_days_labels = []
        last_7_days_total = []
        last_7_days_vehicles = []
        last_7_days_people = []
        last_7_days_plates = []
        
        for i in range(6, -1, -1):
            day = today - timedelta(days=i)
            last_7_days.append(day)
            last_7_days_labels.append(day.strftime('%b %d'))
            
            # Count detections for this day
            day_total = Detection.objects.filter(timestamp__date=day).count()
            day_vehicles = Detection.objects.filter(
                (Q(label='car_no_plate') | Q(label__startswith='car_plate_')) & 
                Q(timestamp__date=day)
            ).count()
            day_people = Detection.objects.filter(
                label='person', 
                timestamp__date=day
            ).count()
            day_plates = Detection.objects.filter(
                label__startswith='car_plate_', 
                timestamp__date=day
            ).count()
            
            last_7_days_total.append(day_total)
            last_7_days_vehicles.append(day_vehicles)
            last_7_days_people.append(day_people)
            last_7_days_plates.append(day_plates)
        
        # Convert to JSON for JavaScript
        context['last_7_days_labels'] = json.dumps(last_7_days_labels)
        context['last_7_days_total'] = json.dumps(last_7_days_total)
        context['last_7_days_vehicles'] = json.dumps(last_7_days_vehicles)
        context['last_7_days_people'] = json.dumps(last_7_days_people)
        context['last_7_days_plates'] = json.dumps(last_7_days_plates)
        
        # Get latest detections
        context['latest_detections'] = Detection.objects.all()[:8]  # Show 8 latest detections
        
        return context
    
    def _calculate_variation(self, current, previous):
        """Calculate percentage variation between two values"""
        if previous == 0:
            return 100 if current > 0 else 0
        return round(((current - previous) / previous) * 100)


def latest_feeds(request):
    """
    Retrieves the latest detection for each distinct camera,
    using a database-agnostic method.
    """
    # Fetch all detections, ordered by camera and then newest timestamp
    all_detections = Detection.objects.order_by('camera', '-timestamp')

    latest_detections_dict = {}
    for detection in all_detections:
        # If the camera is not yet in our dictionary, add the detection.
        # Since the list is sorted by timestamp descending, the first one
        # we encounter for each camera will be the latest.
        if detection.camera not in latest_detections_dict:
            latest_detections_dict[detection.camera] = detection

    latest_detections = list(latest_detections_dict.values())

    context = {
        'latest_detections': latest_detections,
    }

    return render(request, 'viewer/latest_feeds.html', context)