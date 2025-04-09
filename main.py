import time, cv2
from djitellopy import Tello
from threading import Thread
import keyboard
from manual_controls import ManualControls

tello = Tello(retry_count=10)
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()    

tello.takeoff()

kill_switch = ManualControls(tello)
keyboard.hook(kill_switch.on_key_event)

def movements():
    tello.move_up(100)
    tello.rotate_counter_clockwise(360)
    tello.land()

movement_thread = Thread(target=movements)
movement_thread.start()

while True:
    cv2.imshow('frame', frame_read.frame)
    if cv2.waitKey(1) == ord('q'):
        break

movement_thread.join()
