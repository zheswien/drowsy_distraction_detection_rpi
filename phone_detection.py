# import packages
import os

# CV related packages
import cv2
import numpy as np

from tflite_runtime.interpreter import Interpreter
# from tensorflow.lite.python.interpreter import Interpreter
import argparse
from alert import Alert

class PhoneDetection():
    def __init__(self):

        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        self.args = self.argparse()

        self.MODEL_NAME = self.args.modeldir
        self.GRAPH_NAME = self.args.graph
        self.LABELMAP_NAME = self.args.labels
        self.min_conf_threshold = float(self.args.threshold)
        self.resW, self.resH = self.args.resolution.split('x')
        self.imW, self.imH = int(self.resW), int(self.resH)

        # Get path to current working directory
        self.CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, self.MODEL_NAME, self.GRAPH_NAME)

        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, self.MODEL_NAME, self.LABELMAP_NAME)

        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.labels[0] == '???':
            del (self.labels[0])

        # Load the Tensorflow Lite model.
        self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        self.alert = Alert()

        self.phone_exist = "Phone : No"
        self.phone_text_color = (0, 0, 0)


    def argparse(self):
        parser = argparse.ArgumentParser()
        # parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
        #                     required=True)
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                            default='Model')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                            default='ssd_mobiledet_cpu_coco.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                            default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                            default=0.55)
        parser.add_argument('--resolution',
                            help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                            default='400x350')
        return parser.parse_args()

    def phone_main(self,  frame):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        cv2.putText(frame, self.phone_exist, (10, 95), self.font, 0.6, self.phone_text_color, 1, cv2.LINE_AA)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * self.imH)))
                xmin = int(max(1, (boxes[i][1] * self.imW)))
                ymax = int(min(self.imH, (boxes[i][2] * self.imH)))
                xmax = int(min(self.imW, (boxes[i][3] * self.imW)))

                # Draw label
                object_name = self.labels[int(classes[i])]  # Look up object name from "labels" array using class index
                if object_name == "cell phone":
                    self.phone_exist = "Phone : Yes"
                    self.phone_text_color = (0, 0, 255)
                    print("Alert: Distraction Detected, Mobile phone detected, Confidence score: ", int(scores[i] * 100), "%")
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                    label = '%s: %d%%' % ("mobile phone", int(scores[i] * 100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                                  cv2.FILLED)  # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0),
                                1)  # Draw label text
                    self.alert.alert_text(2, frame)
                    self.alert.play_alarm()
                else:
                    self.phone_exist = "Phone : No"
                    self.phone_text_color = (0, 0, 0)