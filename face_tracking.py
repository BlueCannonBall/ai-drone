######### This is a test file for face tracking

import cv2
import numpy as np
from djitellopy import tello
import time

me = tello.Tello()
me.connect()

me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 35, 0)
time.sleep(2.2)

w, h = 360, 240
fbrange = [6000,10000]
pid = [0.4, 0.4, 0]
pError = 0

def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFacesListC = []
    myFaceListArea = []

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w*h
        cv2.circle(img, (cx, cy), 8, (0,255, 0), cv2.FILLED)
        myFacesListC.append([cx, cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFacesListC[i], myFaceListArea[i]]
    else:
        return img, [[0,0], 0]
    
def trackface(me, info, w, pid, pError):
    area = info[1]
    fb = 0
    x, y = info[0]

    error = x - w//2

    speed = pid[0]*error + pid[1]*(error - pError)
    speed = int(np.clip(speed, -100, 100))

    if area > fbrange[0] and area < fbrange[1]:
        fb = 0
    elif area >  fbrange[1]:
        fb = -50
    elif area < fbrange[0] and area != 0:
        fb = 50

    if x == 0:
        speed = 0
        error = 0

    me.send_rc_control(0, fb, 0, speed)

#cap = cv2.VideoCapture(0)
while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img, info = findFace(img)
    img = cv2.resize(img, (w,h))
    trackface(me, info, w , pid, pError)
    print("Area", info[1])
    print("Center", info[0])
    cv2.imshow("testrun", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break