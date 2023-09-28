import os
import cv2
import time
import torch
import threading
import multiprocessing
import numpy as np

from PIL import Image
from logger import logger_gender_recognition
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class GenderRecognition(multiprocessing.Process):
    def __init__(self, id, face_box_queue, gender_queue_to_age_queue, end_signal, lock):
        super().__init__()
        self.id = id
        self.face_box_queue = face_box_queue
        self.gender_queue_to_age_queue = gender_queue_to_age_queue
        self.end_signal = end_signal
        self.lock = lock

        self.device = None
        self.model = None
        self.processor = None

        self.process_times = []

        self.check_video = {} # !!!

    def run(self):
        logger_gender_recognition.info(f"[GenderRecognition_{self.id}] start")

        self.end_signal.value += 1

        self.device = torch.device("cuda:1")  
        self.model = torch.load("gender-classification-2/gender-classification-2_model.pt").to(self.device)
        self.processor = torch.load("gender-classification-2/gender-classification-2_extractor")

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
                    logger_gender_recognition.info(f"[GenderRecognition_{self.id}] get signal -1")
                self.end_signal.value -= 1
                if self.end_signal.value == 0:
                    self.gender_queue_to_age_queue.put(request)
                    # logger_gender_recognition.info(f"[AgeRecognition_{self.id}] process times: {self.process_times}")
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
            gender = preds.item()
            gender = self.model.config.id2label[gender]

            request_copy.text_data.append(f"{gender} ")
            

        self.gender_queue_to_age_queue.put(request_copy)
