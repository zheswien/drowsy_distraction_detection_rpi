# import packages
# CV related packages
import cv2
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from alert import Alert

# Alarm related packages
import time

class FaceLandmark():
    def __init__(self):

        # Facial Landmarks
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
        self.head = [2, 30, 14]

        # Aspesct Ratio
        self.ear = 0
        self.mar = 0
        self.hyar = 0

        # Threshold
        self.eye_thresh = 0.25
        self.mouth_thresh = 0.5
        self.hyar_thresh_up = 0.7
        self.hyar_thresh_down = 0.35

        # Alarm threshold
        self.drowsy_thresh = 0.6
        self.distract_thresh = 2.5
        self.no_face_thresh = 2

        # Font colors
        self.default_color = (0, 0, 0)
        self.alert_color = (0, 0, 255)
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.ear_color = self.default_color
        self.mar_color = self.default_color
        self.hyar_color = self.default_color
        self.drowsy_color = self.default_color
        self.distract_color = self.default_color
        self.noface_color = self.default_color

        # Alarms related variables
        self.ALARM_ON_0 = False
        self.ALARM_ON_1 = False
        self.ALARM_ON_2 = False
        self.drowsy_flag = True
        self.distract_flag = True
        self.noface_flag = True
        self.drowsy_time = 0
        self.distract_time = 0
        self.noface_time = 0

        # Graph plotting variables
        self.start_time = time.time()
        self.seconds = 0.0

        self.alert = Alert()

    # Calculate Eye Aspect Ratio
    def calc_EAR(self, eye, frame):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # Calculate Mouth Aspect Ratio
    def calc_MAR(self, mouth, frame):
        A = dist.euclidean(mouth[1], mouth[7])
        B = dist.euclidean(mouth[2], mouth[6])
        C = dist.euclidean(mouth[3], mouth[5])
        D = dist.euclidean(mouth[0], mouth[4])
        mar = (A + B + C) / (3 * D)
        return mar

    # Calculate Head Yaw Aspect Ratio
    def calc_HYAR(self, head):
        A = dist.euclidean(head[0], head[1])
        B = dist.euclidean(head[0], head[2])
        hyar = A / B
        return hyar

    def reset_param(self, alarm, flag, time):
        alarm = False
        flag = True
        time = 0
        return alarm, flag, time

    def plot_coor(self, coor, frame):
        rows, cols = np.shape(coor)
        for i in range(rows):
            cv2.circle(frame, (coor[i][0], coor[i][1]), 1, (0, 255, 0), -1)

    def face_main(self, faces, predictor, gray, frame):

        # Display Aspect Ratio on screen
        ear_text = "EAR   : {:.2f}".format(self.ear)
        mar_text = "MAR   : {:.2f}".format(self.mar)
        hyar_text = "HYAR  : {:.2f}".format(self.hyar)
        if self.drowsy_time == 0:
            drowsy_time_text = "Drowsy  : 0.0 sec"
        else:
            drowsy_time_text = "Drowsy  : {:.2f} sec".format(time.time() - self.drowsy_time)

        if self.distract_time == 0:
            distract_time_text = "Distract : 0.0 sec"
        else:
            distract_time_text = "Distract : {:.2f} sec".format(time.time() - self.distract_time)
        if self.noface_time == 0:
            noface_time_text = "No face  : 0.0 sec"
        else:
            noface_time_text = "No face  : {:.2f} sec".format(time.time() - self.noface_time)

        cv2.putText(frame, ear_text, (10, 50), self.font, 0.6, self.ear_color, 1, cv2.LINE_AA)
        cv2.putText(frame, mar_text, (10, 65), self.font, 0.6, self.mar_color, 1, cv2.LINE_AA)
        cv2.putText(frame, hyar_text, (10, 80), self.font, 0.6, self.hyar_color, 1, cv2.LINE_AA)
        cv2.putText(frame, drowsy_time_text, (142, 50), self.font, 0.6, self.drowsy_color, 1, cv2.LINE_AA)
        cv2.putText(frame, distract_time_text, (142, 65), self.font, 0.6, self.distract_color, 1, cv2.LINE_AA)
        cv2.putText(frame, noface_time_text, (142, 80), self.font, 0.6, self.noface_color, 1, cv2.LINE_AA)

        if len(faces) == 0:
            self.ear, self.mar, self.hyar = 0, 0, 0
            if self.noface_flag:
                self.noface_time = time.time()
                self.noface_flag = False
            if time.time() - self.noface_time >= self.no_face_thresh:
                print("Alert: No Face Detected, No face time: {:.2f}".format(time.time() - self.noface_time))
                self.noface_color = self.alert_color
                self.alert.alert_text(3, frame)
                self.alert.play_alarm()
                if not self.ALARM_ON_0:
                    self.ALARM_ON_0 = True
                    self.alert.play_alarm()
            if time.time() - self.noface_time >= self.no_face_thresh + 2:
                self.ALARM_ON_0, self.noface_flag, self.noface_time = self.reset_param(self.ALARM_ON_0,
                                                                                       self.noface_flag,
                                                                                       self.noface_time)
        elif len(faces) > 1:
            self.alert.alert_text(4, frame)
            if not self.ALARM_ON_0:
                self.ALARM_ON_0 = True
                self.alert.play_alarm()
        else:
            self.noface_color = self.default_color
            self.ALARM_ON_0, self.noface_flag, self.noface_time = self.reset_param(self.ALARM_ON_0, self.noface_flag,
                                                                                   self.noface_time)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # Get head, eye, mouth coordinates
            head_coor = landmarks[self.head]
            leye_coor = landmarks[self.lStart:self.lEnd]
            reye_coor = landmarks[self.rStart:self.rEnd]
            mouth_coor = landmarks[self.mStart:self.mEnd]

            # Plot coordinates
            self.plot_coor(head_coor, frame)
            self.plot_coor(leye_coor, frame)
            self.plot_coor(reye_coor, frame)
            self.plot_coor(mouth_coor, frame)

            # Calculate Aspect Ratio
            left_ear = self.calc_EAR(leye_coor, frame)
            right_ear = self.calc_EAR(reye_coor, frame)

            self.hyar = self.calc_HYAR(head_coor)
            self.ear = (left_ear + right_ear) / 2
            self.mar = self.calc_MAR(mouth_coor, frame)

            # print("ear: ", self.ear, " mar: ", self.mar, " hyar: ", self.hyar)
            if self.ear < self.eye_thresh or self.mar > self.mouth_thresh:
                if self.ear < self.eye_thresh:
                    self.ear_color = self.alert_color
                else:
                    self.ear_color = self.default_color
                if self.mar > self.mouth_thresh:
                    self.mar_color = self.alert_color
                else:
                    self.mar_color = self.default_color
                if self.drowsy_flag:
                    self.drowsy_time = time.time()
                    self.drowsy_flag = False
                if time.time() - self.drowsy_time > self.drowsy_thresh:
                    print("Alert: Drowsy Detected, EAR: {:.2f}, MAR: {:.2f}, Drowsy time: {:.2f}".format(self.ear, self.mar, time.time() - self.drowsy_time))
                    self.alert.alert_text(1, frame)
                    self.drowsy_color = self.alert_color
                    if not self.ALARM_ON_1:
                        self.ALARM_ON_1 = True
                        self.alert.play_alarm()
                if time.time() - self.drowsy_time >= self.drowsy_thresh + 2:
                    self.ALARM_ON_1, self.drowsy_flag, self.drowsy_time = self.reset_param(self.ALARM_ON_1,
                                                                                           self.drowsy_flag,
                                                                                           self.drowsy_time)
            else:
                self.ALARM_ON_1, self.drowsy_flag, self.drowsy_time = self.reset_param(self.ALARM_ON_1,
                                                                                       self.drowsy_flag,
                                                                                       self.drowsy_time)
                self.ear_color = self.default_color
                self.mar_color = self.default_color
                self.drowsy_color = self.default_color
                self.drowsy_thresh = 0.6

            if self.hyar < self.hyar_thresh_down or self.hyar > self.hyar_thresh_up:
                self.hyar_color = self.alert_color
                if self.distract_flag:
                    self.distract_time = time.time()
                    self.distract_flag = False
                if time.time() - self.distract_time >= self.distract_thresh:
                    print("Alert: Distraction Detected, HYAR: {:.2f}, Distraction time: {:.2f}".format(self.hyar, time.time() - self.distract_time))
                    self.alert.alert_text(2, frame)
                    self.distract_color = self.alert_color
                    if not self.ALARM_ON_2:
                        self.ALARM_ON_2 = True
                        self.alert.play_alarm()
                if time.time() - self.distract_time >= self.distract_thresh + 2:
                    self.ALARM_ON_2, self.distract_flag, self.distract_time = self.reset_param(self.ALARM_ON_2,
                                                                                               self.distract_flag,
                                                                                               self.distract_time)
            else:
                self.ALARM_ON_2, self.distract_flag, self.distract_time = self.reset_param(self.ALARM_ON_2,
                                                                                           self.distract_flag,
                                                                                           self.distract_time)
                self.hyar_color = self.default_color
                self.distract_color = self.default_color

