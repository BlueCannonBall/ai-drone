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

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='detector.tflite'),
    running_mode=VisionTaskRunningMode.VIDEO,
)
detector = FaceDetector.create_from_options(options)

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

def findFace(bgr_image, timestamp):
    largest_face_info = [[0, 0], 0]
    max_area = 0
    output_image = bgr_image.copy() if bgr_image is not None else None

    #rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_image)
    detection_result = detector.detect_for_video(mp_image, int(timestamp))

    for detection in detection_result.detections:
        info = [
            [
                detection.bounding_box.origin_x + detection.bounding_box.width // 2,
                detection.bounding_box.origin_y + detection.bounding_box.height // 2
            ],
            detection.bounding_box.width * detection.bounding_box.height
        ]
        if info[1] > largest_face_info[1]:
            largest_face_info = info
        cv2.rectangle(
            output_image,
            (detection.bounding_box.origin_x, detection.bounding_box.origin_y),
            (detection.bounding_box.origin_x + detection.bounding_box.width, detection.bounding_box.origin_y + detection.bounding_box.height),
            (0, 255, 0),
            2
        )


    return output_image, largest_face_info   

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
    

start = time.time()
last_timestamp = time.monotonic()

# A loop to run the methods
while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w,h))

    now = time.monotonic()
    timestamp = (now - start) * 1000
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
    cv2.imshow("testrun", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break
