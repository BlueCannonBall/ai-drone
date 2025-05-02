import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1)
mp_draw = mp.solutions.drawing_utils

def classify_gestures(landmarks):
    gesture  = ""
    if landmarks[12].y < landmarks[8].y:
        gesture = "go up"
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
    