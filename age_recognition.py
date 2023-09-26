import os
import cv2
import time
import torch
import threading
import multiprocessing
import numpy as np

from PIL import Image
from logger import logger_age_recognition
from transformers import YolosImageProcessor, YolosForObjectDetection
from transformers import ViTForImageClassification, ViTImageProcessor

class AgeRecognition(multiprocessing.Process):
    def __init__(self, id, face_box_queue, age_queue_to_expression_queue, end_signal, lock):
        super().__init__()
        self.id = id
        self.face_box_queue = face_box_queue
        self.age_queue_to_expression_queue = age_queue_to_expression_queue
        self.end_signal = end_signal
        self.lock = lock

        self.device = None
        self.model = None
        self.processor = None

        self.process_times = []

        self.check_video = {} # !!!

        # self.age_dict = {
        #     0: "0 ~ 10  ",
        #     1: "11 ~ 20  ",
        #     2: "21 ~ 30  ",
        #     3: "31 ~ 40  ",
        #     4: "41 ~ 50  ",
        #     5: "51 ~ 60  ",
        #     6: "61 ~ 70  ",
        #     7: "71 ~ 80  ",
        #     8: "81 ~ 90  ",
        #     9: "91 ~ 100  "
        # }

    def run(self):
        logger_age_recognition.info(f"[AgeRecognition_{self.id}] start")

        self.end_signal.value += 1

        self.device = torch.device("cuda:1")
        # self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').to(self.device)
        # self.processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        # self.model = torch.load('yolos-tiny/yolos-tiny_model.pt').to(self.device)
        # self.processor = torch.load('yolos-tiny/yolos-tiny_image_processor')
        # self.model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier').to(self.device)
        # self.processor = ViTImageProcessor.from_pretrained('nateraw/vit-age-classifier')      
        self.model = torch.load('vit-age-classifier/vit-age-classifier_model.pt').to(self.device)
        self.processor = torch.load('vit-age-classifier/vit-age-classifier_image_processor')  

        while True:
            with self.lock:
                request = self.face_box_queue.get()
                # print(f"request.ids: {request.ids}")
                # print(f"request.text_data: {request.text_data}")
                # print(f"request.boxes_data: {request.boxes_data}")

            if request.signal == -1:
                self.face_box_queue.put(request)
                with self.lock:
                    # print(f"[AgeRecognition_{self.id}] get signal -1")
                    logger_age_recognition.info(f"[AgeRecognition_{self.id}] get signal -1")
                self.end_signal.value -= 1
                if self.end_signal.value == 0:
                    self.age_queue_to_expression_queue.put(request)
                    # logger_age_recognition.info(f"[AgeRecognition_{self.id}] process times: {self.process_times}")
                    # print(f"[AgeRecognition_{self.id}] check_video: {self.check_video}") # !!!
                    # logger_person_recognition.info(f"[AgeRecognition_{self.id}] check_video: {self.check_video}") # !!!
                break
            process_start_time = time.time()
            self.process_frame(request)
            process_end_time = time.time()
            self.process_times.append(round(process_end_time - process_start_time, 3))

        # with self.lock:
            # logger_person_recognition.info(f"[PersonRecognition_{self.id}] process times: {self.process_times}")

    def process_frame(self, request):
        # print(f"[AgeRecognition_{self.id}] process_face_box")
        video_id, frame_id, frame_data, boxes_data = request.ids[0], request.ids[1], request.frame_data, request.boxes_data
        request_copy = request.copy()

        request_copy.text_data = []
        for i in range(len(boxes_data)):
            if boxes_data[i] == []:
                request_copy.text_data.append(None)
                continue
            x, y, w, h = int(boxes_data[i][0]), int(boxes_data[i][1]), int(boxes_data[i][2]), int(boxes_data[i][3])
            cropped_image = frame_data[y:y+h,x:x+w]
            inputs = self.processor(cropped_image, return_tensors='pt')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                output = self.model(**inputs)
            
            # Predicted Class probabilities
            proba = output.logits.softmax(1)

            # Predicted Classes
            preds = proba.argmax()
            age = preds.item()
            age = self.model.config.id2label[age]
            # print(f"[AgeRecognition_{self.id}] age: {age}")
            # result_str = f"{age}"
            request_copy.text_data.append(f"{age} ")
            # print(f"[AgeRecognition_{self.id}] result_str: {request_copy.text_data[-1]}")

            # proba = output.logits.softmax(1)
            # preds = proba.argmax(1)
            # age = preds.item()
            # score = proba.max().item()
            # age = self.age_dict[age]
            # result_str = f"{age}"
            # request_copy.text_data.append(result_str)
            # print(f"[AgeRecognition_{self.id}] result_str: {result_str}")
        # print("Before print request")
        self.age_queue_to_expression_queue.put(request_copy)


# gray = cv2.cvtColor(frame_data[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
# id_, confidence = self.recognizer.predict(gray)
# if confidence < 100:
#     emotion = self.names[id_-1]
#     request_copy.text_data.append(emotion)
# else:
#     request_copy.text_data.append(None)

# 1 - 0 ~ 10
# 2 - 11 ~ 20
# 3 - 21 ~ 30
# 4 - 31 ~ 40
# 5 - 41 ~ 50
# 6 - 51 ~ 60
# 7 - 61 ~ 70
# 8 - 71 ~ 80
# 9 - 81 ~ 90
# 10 - 91 ~ 100
