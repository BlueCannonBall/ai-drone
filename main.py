
######### This is a test file for face tracking + hand tracking

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceDetector
from mediapipe.tasks.python.vision.face_detector import FaceDetectorOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python import BaseOptions
import cv2
import numpy as np
from djitellopy import tello
import time
from pid import PID

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.VIDEO
)
detector = PoseLandmarker.create_from_options(options)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1)

me = tello.Tello()
me.connect()

me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 37, 0)
time.sleep(2.2)

w, h = 360, 240
fbrange = [1000,4000]
yaw_pid_controller = PID()
vertical_pid_controller = PID()
horizontal_pid_controller = PID(proportion=0.05, integral=0.0, derivative=0.05)
height = 0

def findFace(rgb_image, timestamp):
    output_image = rgb_image.copy() if rgb_image is not None else None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect_for_video(mp_image, timestamp)

    if len(detection_result.pose_landmarks) == 0:
        return output_image, [[0, 0], 0]
    height, width, channels = rgb_image.shape
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

def move_drone(me, gesture, height):
    if gesture == "Go up":
        height = me.get_height()
        if (height <= 185):
            me.send_rc_control(0,0,30,0)
        else:
            me.send_rc_control(0,0,0,0)
            time.sleep(1)
            me.flip('b')
            time.sleep(3)
            me.send_rc_control(0,0,0,0)
            time.sleep(1)

    if gesture == "Go down":
        me.send_rc_control(0,0,-30,0)
    if gesture == "Go right":
        me.send_rc_control(30,0,0,0)
    if gesture == "Go left":
        me.send_rc_control(-30,0,0,0)

    # need to include PID controls 
    return

def trackface(me, info, w, yaw_pid_controller, vertical_pid_controller, horizontal_pid_controller):
    # Gets informations from findFace()
    area = info[1]
    fb = 0
    x, y = info[0]

    # Error from the center of the screen
    yaw_error = x - (w//2)

    # PID Calculations to change yaw of the drone according to the Error
    yaw_speed = yaw_pid_controller.calculate(yaw_error)
    yaw_speed = int(np.clip(yaw_speed, -100, 100))
    

    # Moves the drone forward or backward based on the area of the rectangle on the face
    xError = 2500 -area
    horizontal_speed = horizontal_pid_controller.calculate(xError)
    horizontal_speed = int(np.clip(horizontal_speed, -35, 35))

    yError = -(y - (h // 2.5))
    vertical_speed = vertical_pid_controller.calculate(yError)
    vertical_speed = int(np.clip(vertical_speed, -35, 35))
    
    # if yError > 40:
    #     vertical_speed = -30
    # if yError < -40:
    #     vertical_speed = 30

    # if area > fbrange[1]:
    #     vertical_speed = 0

    if x == 0:
        vertical_speed = 0
        yaw_speed = 0
        horizontal_speed = 0
        
    me.send_rc_control(0, horizontal_speed, vertical_speed, yaw_speed)
    
    

start = time.monotonic()
last_timestamp = 0

# A loop to run the methods
while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w,h))

    results = hands.process(img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = img.shape
            #height = me.get_height()
            # mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            middle_finger = hand_landmarks.landmark[12]
            index_finger = hand_landmarks.landmark[8]
            wrist = hand_landmarks.landmark[0]

            middle_finger_coords = (int(middle_finger.x*w), int(middle_finger.y*h))
            index_finger_coords = (int(index_finger.x*w), int(index_finger.y*h))
            wrist_coords = (int(wrist.x*w), int(wrist.y*h))

            cv2.circle(img, middle_finger_coords, 4, (0, 225, 0), 2)
            cv2.circle(img, index_finger_coords, 4, (0, 225, 0), 2)
            cv2.circle(img, wrist_coords, 4, (0, 225, 0), 2)

            gesture = classify_gestures(hand_landmarks.landmark)
            print(gesture)
            #print(height)
            move_drone(me, gesture, height)
    else:
        now = time.monotonic()
        timestamp = int((now - start) * 1000)
        # mediapipe expects a strictly increasing timestamp
        # this bit of code ensures that the last timestamp is never equal to the new one
        # t_n < t_n+1
        if timestamp == last_timestamp:
            timestamp += 1
        last_timestamp = timestamp
        img, info = findFace(img, timestamp)
        if info[1] != 0:
            trackface(me, info, w , yaw_pid_controller, vertical_pid_controller, horizontal_pid_controller)
            print("Area", info[1])
            print("Center", info[0])
        else:
            # No faces found, spin in a circle to find some:
            me.send_rc_control(0, 0, 0, 30)


    #print (error) Test condition
    bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("testrun", bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break
