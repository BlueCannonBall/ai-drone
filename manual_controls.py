class ManualControls:
    def __init__(self, tello):
        self.tello = tello

    def on_key_event(self, e):
        if e.name == "k":
            self.tello.land()
