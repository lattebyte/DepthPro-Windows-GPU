# Quick solutions during test, not recommended for production
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image
import torch
import cv2
import numpy as np
from torch.cuda.amp import autocast
from lattebyte_cm16a_camera import initialize_camera, configure_camera
from depthpro_gpu_setting import configure_gpu


def main():
    # Initialize the camera
    lattebyte_cm16a = initialize_camera()

    # Configure camera settings
    configure_camera(lattebyte_cm16a)
    
    # Configure yolo settings
    model,transform, device = configure_gpu()
    
    while True:
        # Read a frame from the video
        ret, frame = lattebyte_cm16a.read()
        
        if not ret:
            break

        # Reduce frame size to speed up processing (optional)
        frame = cv2.resize(frame, (1280, 720))  # Lower resolution for faster processing

        # Convert the OpenCV frame (BGR) to a PIL image (RGB)
        image_x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_x)

        # Load and preprocess the image
        image = transform(pil_image)
        
        # Ensure the image has a batch dimension if needed by the model
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension if necessary

        # Move the image to GPU (if available)
        image = image.to(device)

        # Run inference with mixed precision
        with torch.no_grad():
            with autocast():
                prediction = model.infer(image)

        # Extract depth map (in meters)
        depth = prediction["depth"]

        # Convert depth tensor to numpy for visualization
        depth_np = depth.squeeze().cpu().numpy()

        # Normalize depth values for better visualization
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to an 8-bit image for display
        depth_8bit = np.uint8(depth_norm)

        # Apply a colormap for visualization
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_PLASMA)

        # Display depth map using OpenCV
        cv2.imshow('Inferred Depth Map', depth_colormap)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the windows
    lattebyte_cm16a.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()