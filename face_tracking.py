######### This is a test file for face tracking

import mediapipe as mp
import cv2
import numpy as np
from djitellopy import tello
import time

me = tello.Tello()
me.connect()

me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 37, 0)
time.sleep(2.2)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

w, h = 360, 240
fbrange = [2000,8000]
pid = [0.5, 0.5, 0.1]
pError = 0
dt = 0.1
integral = 0

def findFace(bgr_image):
    largest_face_info = [[0, 0], 0]
    max_area = 0
    output_image = bgr_image.copy() if bgr_image is not None else None

    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:

        if bgr_image is None:
            return bgr_image, largest_face_info

        image_height, image_width, _ = bgr_image.shape
        results = face_detection.process(bgr_image)

        largest_bbox = None

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                if bboxC: 
                    xmin = int(bboxC.xmin * image_width)
                    ymin = int(bboxC.ymin * image_height)
                    width = int(bboxC.width * image_width)
                    height = int(bboxC.height * image_height)

                    if width > 0 and height > 0:
                         current_area = width * height
                         if current_area > max_area:
                             max_area = current_area
                             center_x = xmin + width // 2
                             center_y = ymin + height // 2
                             largest_face_info = [[center_x, center_y], max_area]
                             largest_bbox = (xmin, ymin, width, height)

    if largest_bbox:
            x, y, w, h = largest_bbox
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
    

# A loop to run the methods
while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w,h))

    img, info = findFace(img)
    pError = trackface(me, info, w , pid, pError)
    print("Area", info[1])
    print("Center", info[0])
    #print (error) Test condition
    cv2.imshow("testrun", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break
