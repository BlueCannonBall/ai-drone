import time, cv2
from djitellopy import Tello
from threading import Thread
import keyboard
from manual_controls import ManualControls

tello = Tello(retry_count=8)
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()    

tello.takeoff()

kill_switch = ManualControls(tello)
keyboard.hook(kill_switch.on_key_event)

def movements():
    tello.move_up(50)
    time.sleep(3)
    tello.rotate_counter_clockwise(360)
    time.sleep(3)
    tello.land()

movement_thread = Thread(target=movements)
movement_thread.start()

def battery_percent():
    while True:
        if (tello.get_battery() < 20):
            tello.land()
            print("Low Battery, needs to be changed")
            break
        time.sleep(5) # Check the battery every 5 seconds
    
battery_thread = Thread(target=battery_percent, daemon=True)
battery_thread.start()

while True:
    frame_bgr = cv2.cvtColor(frame_read.frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', frame_bgr)
    if cv2.waitKey(1) == ord('q'):
        break

movement_thread.join()
