
######### This is a test file for face tracking

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

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.VIDEO
)
detector = PoseLandmarker.create_from_options(options)

me = tello.Tello()
me.connect()

me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 37, 0)
time.sleep(2.2)

w, h = 360, 240
fbrange = [2000,8000]
pid = [0.5, 0.5, 0.1]
pError = 0
dt = 0.1
integral = 0

def findFace(rgb_image, timestamp):
    output_image = rgb_image.copy() if rgb_image is not None else None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect_for_video(mp_image, timestamp)

    right_eye = detection_result.pose_world_landmarks[0][5]
    left_eye = detection_result.pose_world_landmarks[0][2]
    eye_distance = abs(right_eye.x - left_eye.x)
    area = eye_distance * eye_distance
    cx = left_eye.x + (right_eye.x - left_eye.x) / 2
    cy = right_eye.y
    cv2.rectangle(
        output_image,
        (left_eye.x, left_eye.y + eye_distance / 2),
        (right_eye.x, right_eye.y - eye_distance / 2),
        (0, 255, 0),
        2
    )

    return output_image, [[cx, cy], area]

def trackface(me, info, w, pid, pError):
    # Gets informations from findFace()
    area = info[1]
    fb = 0
    x, y = info[0]

    # Error from the center of the screen
    error = x - (w//2)

    # PID Calculations to change yaw of the drone according to the Error
    integral = error*dt
    derivative = (error-pError) / dt if dt > 0 else 0
    pError = error

    speed = pid[0]*error + pid[1]*(integral) + pid[2]*derivative
    speed = int(np.clip(speed, -100, 100))
    

    # Moves the drone forward or backward based on the area of the rectangle on the face
    if area > fbrange[0] and area < fbrange[1]:
        fb = 0
    elif area >  fbrange[1]:
        fb = -40
    elif area < fbrange[0] and area != 0:
        fb = 40

    if x == 0:
        speed = 0
        error = 0
        
    me.send_rc_control(0, fb, 0, speed)

    '''
    if error > 5:
        speed = -30
        me.send_rc_control(0, fb, 0, speed)
    elif error < -5:
        speed = 30
        me.send_rc_control(0, fb, 0, speed)
    '''
    
    return pError
    

start = time.monotonic()
last_timestamp = 0

# A loop to run the methods
while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w,h))

    now = time.monotonic()
    timestamp = int((now - start) * 1000)
    # mediapipe expects a strictly increasing timestamp
    # this bit of code ensures that the last timestamp is never equal to the new one
    # t_n < t_n+1
    if timestamp == last_timestamp:
        timestamp += 1
    last_timestamp = timestamp
    img, info = findFace(img, timestamp)
    pError = trackface(me, info, w , pid, pError)
    print("Area", info[1])
    print("Center", info[0])
    #print (error) Test condition
    bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("testrun", bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break
