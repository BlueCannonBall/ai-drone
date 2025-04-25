import cv2
import mediapipe as mp
import math

#initializing face_mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

#Drawing utilities
mp_drawing = mp.solutions.drawing_utils

#Geometric distance between two points
def distance_between_points(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

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
            print ("Distance between eyes (pixels): ", distance)

    cv2.imshow("Eye distance", frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break

cap.release()
cap.destroyAllWindows()



        