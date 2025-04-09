import keyboard
import time, cv2
from threading import Thread
from djitellopy import Tello

class Manual_controls:
    def __init__(self, name):
         self.name = name

    def on_key_event(self, e):
            if e.event_type == keyboard.KEY_DOWN:
                return(e.name)

    def run_program():
        keyboard.hook(on_key_event(e))
        
        print("Start typing...")
        keyboard.wait()