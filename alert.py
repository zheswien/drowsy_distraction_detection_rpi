import cv2
# from pygame import mixer  # Uncomment to use bluetooth speaker
import RPi.GPIO as GPIO # Uncomment to use buzzer
from time import sleep

class Alert:
    def __init__(self):
        # Uncomment below if using buzzer
        GPIO.setwarnings(False)  # Disable warnings (optional)
        GPIO.setmode(GPIO.BCM)  # Select GPIO mode
        self.buzzer = 15  # Set buzzer - pin 15 as output
        GPIO.setup(self.buzzer, GPIO.OUT)

        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.text_position = (10,155)
        self.text_position2 = (10,180)

    def alert_text(self, alert_code, frame):
        if alert_code == 1:
            cv2.putText(frame, "Drowsy detected!!", self.text_position, self.font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        elif alert_code == 2:
            cv2.putText(frame, "Distraction detected!!", self.text_position2, self.font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        elif alert_code == 3:
            cv2.putText(frame, "No Face Detected", self.text_position, self.font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        elif alert_code == 4:
            cv2.putText(frame, "Multiple Face Detected", self.text_position, self.font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    def play_alarm(self):
        # Uncomment below to use speaker (Play audio)
        # mixer.init()
        # mixer.music.load('AlarmSound.wav')
        # mixer.music.play()

        # Uncomment below to activate buzzer
        GPIO.output(self.buzzer,GPIO.HIGH)
        sleep(0.3) # Delay in seconds
        GPIO.output(self.buzzer,GPIO.LOW)
        sleep(0.2)
        GPIO.output(self.buzzer, GPIO.HIGH)
        sleep(0.3)  # Delay in seconds
        GPIO.output(self.buzzer, GPIO.LOW)
        sleep(0.2)
        GPIO.output(self.buzzer, GPIO.HIGH)
        sleep(0.3)  # Delay in seconds
        GPIO.output(self.buzzer, GPIO.LOW)
        sleep(0.2)

    def stop_alarm(self):
        GPIO.cleanup()