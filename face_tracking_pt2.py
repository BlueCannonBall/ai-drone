import cv2
import mediapipe as mp
import math
from djitellopy import tello
import time
import numpy as np

me = tello.Tello()
me.connect()

me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 37, 0)
time.sleep(2)

pixel_range = [80,130]
pid = [0.5, 0.5, 0.1]
pError = 0
dt = 0.1
integral = 0

#initializing face_mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1)

#Drawing utilities
mp_drawing = mp.solutions.drawing_utils

#Geometric distance between two points
def distance_between_points(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def track_face(me, pid, pError, distance_between_eyes):
    fb = 0

    # Error from the center of the screen
    error =  distance_between_eyes/2

    # PID Calculations to change yaw of the drone according to the Error
    integral = error*dt
    derivative = (error-pError) / dt if dt > 0 else 0
    pError = error

    speed = pid[0]*error + pid[1]*(integral) + pid[2]*derivative
    speed = int(np.clip(speed, -100, 100))

    if distance_between_eyes < pixel_range[0] and distance_between_eyes > pixel_range[1]:
        fb = 0
    elif distance_between_eyes < pixel_range[0] and distance_between_eyes > 0:
        fb = 40
    elif distance_between_eyes > pixel_range[1]:
        fb = -40

    if distance_between_eyes == 0:
        speed = 0
        error = 0

    me.send_rc_control(0, fb, 0, speed)
    return pError

#cap = cv2.VideoCapture(0)

while True:

    frame = me.get_frame_read().frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    distance = 0
    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:

            #gets h, w of the frame
            h, w, _ = frame.shape

            #gets the coords of left and right eye and converts it into pixels
            left_eye = faceLandmarks.landmark[33]
            right_eye = faceLandmarks.landmark[263]

            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y*h))

            cv2.circle(frame, left_eye_coords, 5, (0, 225, 0), -1)
            cv2.circle(frame, right_eye_coords, 5, (0, 225, 0), -1)

            #computes distance
            distance = distance_between_points(left_eye_coords, right_eye_coords)
    
    pError = track_face(me, pid, pError, distance)
    cv2.imshow("Drone vision", frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break





        