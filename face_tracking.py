######### This is a test file for face tracking

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

w, h = 360, 240
fbrange = [2000,8000]
pid = [0.5, 0.5, 0.1]
pError = 0
dt = 0.1
integral = 0

def findFace(img):
    # Setting up the face tracking model 
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    # A list of center co-ordinates of the face and the area of the reactangle surrounding the face.
    # This changes based on the distance from the drone's camera
    myFacesListC = []
    myFaceListArea = []

    # Calculations for finding the center of the face and 
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w*h
        cv2.circle(img, (cx, cy), 8, (0,255, 0), cv2.FILLED)
        myFacesListC.append([cx, cy])
        myFaceListArea.append(area)

    # Finds the maximum area and returns the area along with the center of that area
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFacesListC[i], myFaceListArea[i]]
    else:
        return img, [[0,0], 0]
    
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