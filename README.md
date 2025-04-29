# ai-drone

Right now, this repository is a collection of disparate scripts for controlling Tello drones. The end goal is to be able to effectively control a Tello drone through hand gestures and body language.

## Usage

Install dependencies using `pip install -r requirements.txt`. Then, connect your device to the drone's Wi-Fi network, and run the program using `python pose_recognition.py`. `pose_recognition.py` uses MediaPipe to find your head and it directs the drone to physically follow your head as it moves.
