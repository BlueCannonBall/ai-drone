import time, cv2
from threading import Thread
from djitellopy import Tello

tello = Tello()

tello.connect()

keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()    

tello.takeoff()

while keepRecording:
    cv2.imshow('frame', frame_read.frame)
    #video.write(frame_read.frame)
    if cv2.waitKey(1) == ord('q'):
        break

tello.move_up(100)
tello.rotate_counter_clockwise(360)
tello.land()

keepRecording = False
