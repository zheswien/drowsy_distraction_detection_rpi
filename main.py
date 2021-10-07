# Import packages
import cv2
import time
import dlib
import argparse
import numpy as np
from VideoStream import VideoStream
from face_landmark import FaceLandmark
from phone_detection import PhoneDetection
from alert import Alert
# from imutils.video import FileVideoStream
# from button import Shutdown
import _thread
import os

def shutdown():
    os.system('python button.py')



parser = argparse.ArgumentParser()
parser.add_argument('--shape_predictor', help='Name of the shape predictor file, if different than shape_predictor_68_face_landmarks',
                    default='shape_predictor_68_face_landmarks.dat')
args = parser.parse_args()


# Load Facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

fl = FaceLandmark()
pd = PhoneDetection()
# shutdown = Shutdown()

# Initialize video stream
videostream = VideoStream(resolution=(pd.imW, pd.imH)).start()
# frame1 = cv2.imread("phone.jpg")
# videostream = cv2.VideoCapture(0)
# videostream = FileVideoStream("output.avi").start()
time.sleep(1)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('newvideo.mp4', fourcc, 30.0, (int(videostream.get(3)),int(videostream.get(4))))
out = cv2.VideoWriter('newvideo.avi', fourcc, 1.4, (400, 350))
_thread.start_new_thread(shutdown, ())

while True:

    # shutdown.shutdown()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # rect, frame1 = videostream.read()
    # frame1 = cv2.resize(frame1, (500, 430), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    # frame = np.array(frame1)

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    cv2.rectangle(frame, (5, 10), (310, 108), (235, 235, 235), -1)
    # Draw framerate in corner of frame
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                (227, 132, 89), 1,
                cv2.LINE_AA)

    fl.face_main(faces, predictor, gray, frame)
    pd.phone_main(frame)

    # Display results on frame
    cv2.imshow('Drowsy & Distraction Detection', frame)
    # print(cv2.getWindowImageRect('Drowsy & Distraction Detection'))

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    out.write(frame)

    key = cv2.waitKey(1)
    if key == 27: # ESC key
        break
    if cv2.getWindowProperty('Drowsy & Distraction Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
Alert.stop_alarm()
out.release()
