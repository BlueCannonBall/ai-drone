import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceDetector
from mediapipe.tasks.python.vision.face_detector import FaceDetectorOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python import BaseOptions

import math
from djitellopy import tello
import time

me = tello.Tello()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1)
mp_draw = mp.solutions.drawing_utils

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.VIDEO
)
detector = PoseLandmarker.create_from_options(options)

def classify_gestures(landmarks):
    gesture = ""
    if landmarks[12].y < landmarks[8].y:
        gesture = "Go up"
    if landmarks[12].y > landmarks[8].y:
        if landmarks[12].x > landmarks[8].x:
            gesture = "Go right"
        if landmarks[12].x < landmarks[8].x:
            gesture = "Go left"
    if landmarks[12].y > landmarks[0].y:
        gesture = "Go down"
    '''else:
        gesture = "No hand Detected"'''
    return gesture

def move_drone(me, gesture):
    if gesture == "Go up":
        me.send_rc_contol(0,0,0,0)
    if gesture == "Go down":
        me.send_rc_contol(0,0,0,0)
    if gesture == "Go right":
        me.send_rc_contol(0,0,0,0)
    if gesture == "Go left":
        me.send_rc_contol(0,0,0,0)

    # need to include PID controls 
    return

# A function to keep a certain distance from the face/hand
# Arguments: OpenCV image, video timestamp, convert (bool)
# Set convert to True if the input image is in BGR format.
# Returns: annotated image, [[center_x, center_y], area]
def distance_maintainer(image, timestamp: int, convert=False):
    output_image = image.copy() if image is not None else None

    if convert:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect_for_video(mp_image, timestamp)

    if len(detection_result.pose_landmarks) == 0:
        return output_image, [[0, 0], 0]
    height, width, channels = image.shape
    width = float(width)
    height = float(height)
    right_eye = detection_result.pose_landmarks[0][5]
    left_eye = detection_result.pose_landmarks[0][2]
    eye_distance = abs(right_eye.x * width - left_eye.x * width) 
    area = eye_distance * eye_distance * 8.0
    cx = left_eye.x * width + (right_eye.x * width - left_eye.x * width) / 2
    # Average of the two y values for the eyes
    cy = (right_eye.y * height + left_eye.y * height) / 2
    cv2.rectangle(
        output_image,
        (int(left_eye.x * width), int(left_eye.y * height + eye_distance / 2)),
        (int(right_eye.x * width), int(right_eye.y * height - eye_distance / 2)),
        (0, 255, 0),
        2
    )
    return output_image, [[cx, cy], area]
 

while True:
    success, img = cap.read()
    rgb_image = img
    rgb_image = cv2.flip(rgb_image, 1)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = rgb_image.shape
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            middle_finger = hand_landmarks.landmark[12]
            index_finger = hand_landmarks.landmark[8]
            wrist = hand_landmarks.landmark[0]

            middle_finger_coords = (int(middle_finger.x*w), int(middle_finger.y*h))
            index_finger_coords = (int(index_finger.x*w), int(index_finger.y*h))
            wrist_coords = (int(wrist.x*w), int(wrist.y*h))

            cv2.circle(rgb_image, middle_finger_coords, 4, (0, 225, 0), 2)
            cv2.circle(rgb_image, index_finger_coords, 4, (0, 225, 0), 2)
            cv2.circle(rgb_image, wrist_coords, 4, (0, 225, 0), 2)

            gesture = classify_gestures(hand_landmarks.landmark)
            print(gesture)

    cv2.imshow("Gesture recognition", rgb_image)

    if cv2.waitKey(1) & 0XFF == 27:
        break
    
