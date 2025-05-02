import cv2
import time

# --- Configuration ---
# TODO: Replace with actual camera credentials and IP address
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "123456"
CAMERA_IP = "10.0.0.115"
CHANNEL = 1  # Usually 1 for the first channel
STREAM_TYPE = 1  # 1 for main stream, 2 for sub stream

RTSP_URL = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMERA_IP}:554/Streaming/channels/{CHANNEL}0{STREAM_TYPE}"

def connect_and_capture(rtsp_url):
    """Connects to the RTSP stream and captures frames."""
    print(f"Attempting to connect to: {rtsp_url.replace(CAMERA_PASSWORD, '********')}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Successfully connected to the camera stream.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to retrieve frame or stream ended.")
                # Attempt to reconnect or handle error
                time.sleep(5) # Wait before retrying
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    print("Error: Failed to reconnect.")
                    break
                else:
                    print("Reconnected successfully.")
                    continue

            # --- Frame Processing Placeholder ---
            # TODO: Add object detection logic here in the next step
            # For now, just display the frame dimensions
            if frame_count % 30 == 0: # Print info every 30 frames
                h, w, _ = frame.shape
                print(f"Frame {frame_count}: Dimensions {w}x{h}")
            # cv2.imshow('Camera Feed', frame) # Uncomment to display feed locally if needed
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # ------------------------------------

            frame_count += 1

            # Optional: Limit capture duration for testing
            # if time.time() - start_time > 60: # Run for 60 seconds
            #     print("Test duration finished.")
            #     break

    except KeyboardInterrupt:
        print("Stream capture stopped by user.")
    finally:
        print("Releasing video capture resource.")
        cap.release()
        # cv2.destroyAllWindows() # Uncomment if using imshow

if __name__ == "__main__":
    connect_and_capture(RTSP_URL)

