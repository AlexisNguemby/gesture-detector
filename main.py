import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as image_module

# Setup hand detector
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
BaseOptions = base_options.BaseOptions
RunningMode = vision.RunningMode
ImageFormat = image_module.ImageFormat

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.VIDEO
)

try:
    landmarker = HandLandmarker.create_from_options(options)
except Exception as e:
    # Fallback to IMAGE mode if VIDEO fails
    print(f"Erreur: {e}")
    print("Utilisation du mode IMAGE à la place...")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task')
    )
    landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe image
    mp_image = image_module.Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands
    hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_count)
    
    # Draw landmarks and connections
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm
            ]
            
            for start_idx, end_idx in connections:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_pos = (int(start.x * w), int(start.y * h))
                end_pos = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_pos, end_pos, (255, 0, 0), 2)
    
    cv2.imshow("Hand Detection with MediaPipe", frame)
    frame_count += 1
    
    # Press Q to quit or close window
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # Check if window was closed
    if cv2.getWindowProperty("Hand Detection with MediaPipe", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()