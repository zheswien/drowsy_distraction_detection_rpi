import RPi.GPIO as GPIO
import time
import os

buttonPin = 10

last_state = True
input_state = True
        

def shutdown():
    GPIO.setwarnings(False)  # Disable warnings (optional)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(buttonPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    input_state = GPIO.input(buttonPin)
    if (not input_state):
        print("Shutdown")
        # os.system('pkill -9 -f button.py')
        # os.system('pkill -9 -f drowsy_distraction_detection\main.py')
        os.system('sudo shutdown -h now')
        time.sleep(0.05)
        GPIO.cleanup()

while True:
    shutdown()
