# Hikvision Object Detection & LPR Django App for Raspberry Pi 5

This application connects to a Hikvision camera stream, performs object detection (person, car) using YOLOv8, performs license plate recognition (LPR) on detected cars using EasyOCR, saves detection images and metadata, and provides a Django web interface to view the live stream and browse detection history.

## Features

*   Connects to Hikvision camera via RTSP.
*   Detects persons and cars using YOLOv8n.
*   Performs LPR on detected cars using EasyOCR.
*   Saves cropped images of detected objects.
*   Stores detection metadata (timestamp, label, image path) in a SQLite database.
*   Django web interface:
    *   Live view of the camera stream (using sub-stream for efficiency).
    *   History page displaying detected objects with images, labels (including plate numbers), and timestamps.
    *   Search functionality on the history page.
    *   Pagination for history browsing.
*   Automatic cleanup of detection records and images older than 60 days (configurable).
*   Designed to run on Raspberry Pi 5 (CPU execution for ML models).

## Project Structure

```
/hikvision_detector/
|-- media/                  # Stores saved detection images
|   |-- detections/
|-- viewer/                 # Django app for viewing detections
|   |-- migrations/
|   |-- management/
|   |   |-- commands/
|   |   |   |-- cleanup_detections.py # Management command for cleanup
|   |-- templates/
|   |   |-- viewer/
|   |   |   |-- base.html
|   |   |   |-- history.html
|   |   |   |-- live.html
|   |-- __init__.py
|   |-- admin.py
|   |-- apps.py
|   |-- models.py           # Database model for detections
|   |-- tests.py
|   |-- urls.py             # URLs for the viewer app
|   |-- views.py            # Views for live stream and history
|-- webapp/                 # Django project settings
|   |-- __init__.py
|   |-- asgi.py
|   |-- settings.py
|   |-- urls.py
|   |-- wsgi.py
|-- .gitignore              # (Recommended to add)
|-- camera_connector.py     # Basic script for testing connection (not used by Django app)
|-- detector_django.py      # Core detection script integrated with Django
|-- manage.py               # Django management script
|-- requirements.txt        # Python dependencies (generate before deployment)
|-- todo.md                 # Development task list
|-- venv/                   # Python virtual environment
|-- yolov8n.pt              # YOLO model (downloaded automatically)
|-- db.sqlite3              # SQLite database file
```

## Setup Instructions (Raspberry Pi 5)

1.  **Prerequisites:**
    *   Raspberry Pi 5 with Raspberry Pi OS (or similar Linux distribution) installed.
    *   Python 3.11.2 installed.
    *   Internet connection (for downloading packages and models).
    *   Ensure your Hikvision camera is connected to the same network and RTSP streaming is enabled.

2.  **Clone/Extract Project:**
    *   Transfer the provided `.zip` file to your Raspberry Pi.
    *   Unzip the archive: `unzip hikvision_detector.zip`
    *   Navigate into the project directory: `cd hikvision_detector`

3.  **Create Virtual Environment:**
    *   `python3 -m venv venv`
    *   Activate the environment: `source venv/bin/activate`

4.  **Install Dependencies:**
    *   **(Important)** Generate `requirements.txt` if not already present or up-to-date:
        ```bash
        # (Inside the activated venv)
        pip freeze > requirements.txt
        ```
    *   Install packages from the requirements file:
        ```bash
        pip install -r requirements.txt
        ```
        *Note: Installation, especially for PyTorch/Torchvision/Ultralytics/EasyOCR, might take a significant amount of time on the RPi.* 
        *If you encounter issues, try installing PyTorch/Torchvision separately first, following official instructions for ARM64/RPi if available, then install the rest.* 
        *Make sure you have sufficient disk space.* 

5.  **Configure Camera Details:**
    *   Edit `detector_django.py` and `viewer/views.py`.
    *   Update the `CAMERA_USERNAME`, `CAMERA_PASSWORD`, and `CAMERA_IP` variables with your actual Hikvision camera credentials and IP address.
    *   Adjust `CHANNEL` and `STREAM_TYPE` if necessary (usually `CHANNEL=1`, `STREAM_TYPE=1` for main stream, `STREAM_TYPE=2` for sub-stream).

6.  **Database Setup:**
    *   Apply database migrations:
        ```bash
        python manage.py migrate
        ```

7.  **Download AI Models (First Run):**
    *   The YOLOv8 and EasyOCR models will be downloaded automatically the first time the detection script or web server runs and accesses them. This might take time.

## Usage

There are two main components to run:

1.  **Detection Script (Background Process):** This script connects to the camera, performs detection/LPR, and saves data to the database.
2.  **Django Web Server:** This serves the web interface for live view and history.

**Running the Detection Script:**

*   It's recommended to run this as a background service using `systemd` or `supervisor` for reliability.
*   **Manual/Testing:**
    ```bash
    # (Ensure virtual environment is active: source venv/bin/activate)
    python detector_django.py
    ```
    *   Leave this running in a terminal or manage it with a tool like `screen` or `tmux`.

**Running the Django Web Server:**

*   **Development Server (for testing):**
    ```bash
    # (Ensure virtual environment is active: source venv/bin/activate)
    # Run on port 8000, accessible on your local network
    python manage.py runserver 0.0.0.0:8000
    ```
    *   Access the web interface from another device on the same network by navigating to `http://<RaspberryPi_IP>:8000` in your browser.
*   **Production Deployment:** For more robust deployment, consider using `gunicorn` and `nginx`.

**Accessing the Web Interface:**

*   **History:** `http://<RaspberryPi_IP>:8000/`
*   **Live View:** `http://<RaspberryPi_IP>:8000/live/`

**Running Data Cleanup:**

*   This command deletes records and images older than 60 days (or the specified number of days).
*   Schedule this to run periodically (e.g., daily) using `cron`.
    ```bash
    # (Ensure virtual environment is active: source venv/bin/activate)
    # Dry run (shows what would be deleted):
    # python manage.py cleanup_detections --days 60 

    # Actual deletion:
    python manage.py cleanup_detections --days 60 
    ```

## Notes & Optimization (RPi5)

*   **Performance:** Object detection (YOLOv8) and LPR (EasyOCR) are computationally intensive. Running on the RPi5 CPU will be significantly slower than on a system with a dedicated GPU. Expect lower frame rates for detection.
*   **Model Choice:** `yolov8n.pt` (Nano) is used for better performance on CPU. You could experiment with even smaller models if needed, potentially sacrificing accuracy.
*   **Sub-stream:** The live view uses the camera's sub-stream (`STREAM_TYPE=2` in `viewer/views.py`) by default for better web performance. The detection script (`detector_django.py`) uses the main stream (`STREAM_TYPE=1`) for higher resolution analysis. Adjust if needed.
*   **Throttling:** The detection script includes basic throttling (`detection_interval`) to avoid saving images of the same object too frequently.
*   **Memory:** Monitor memory usage, especially with multiple processes running.
*   **Heat:** Ensure adequate cooling for the Raspberry Pi 5, as sustained CPU load from detection can generate heat.

