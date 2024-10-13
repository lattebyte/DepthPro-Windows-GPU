import cv2

def initialize_camera():
    """Initialize the camera and return the camera object."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Could not open video device")
    else:
        print("Camera is initialized")
    return camera

def configure_camera(camera):
    """Configure camera properties such as resolution and frame rate."""
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height
    camera.set(cv2.CAP_PROP_FPS, 120)  # Set frames per second
    camera.set(cv2.CAP_PROP_EXPOSURE, -5)  # Adjust this based on your camera's range

    print("Actual camera FPS: ", camera.get(cv2.CAP_PROP_FPS))  # Check actual FPS
    return camera
