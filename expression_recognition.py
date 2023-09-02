import os
import cv2
import time
import torch
import threading
import multiprocessing

import numpy as np
from logger import logger_expression_recognition

class ExpressionRecognition(multiprocessing.Process):
    def __init__(self, id, face_box_queue, expression_message_queue, end_signal, lock):
        super().__init__()
        self.id = id
        self.face_box_queue = face_box_queue
        self.expression_message_queue = expression_message_queue
        self.end_signal = end_signal
        self.lock = lock

        self.device = None
        self.recognizer = None
        self.names = None

        self.process_times = []

    def run(self):
        logger_expression_recognition.info(f"[ExpressionRecognition_{self.id}] start")

        self.end_signal.value += 1

        self.device = torch.device("cuda:0")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainer_emotion.yml')
        self.names = ['angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry',
         'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy',
         'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
         'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad']        

        while True:
            with self.lock:
                request = self.face_box_queue.get()
            if request.signal == -1:
                self.face_box_queue.put(request)
                with self.lock:
                    logger_expression_recognition.info(f"[ExpressionRecognition_{self.id}] get signal -1")
                self.end_signal.value -= 1
                if self.end_signal.value == 0:
                    self.expression_message_queue.put(request)
                break
            process_start_time = time.time()
            self.process_face_box(request)
            process_end_time = time.time()
            self.process_times.append(round(process_end_time - process_start_time, 3))

        # with self.lock:
            # logger_expression_recognition.info(f"[ExpressionRecognition_{self.id}] process times: {self.process_times}")

    def process_face_box(self, request): # TODO: expression recognition
        video_id, frame_id, frame_data, boxes_data = request.ids[0], request.ids[1], request.frame_data, request.boxes_data
        request_copy = request.copy()

        request_copy.text_data = []
        for i in range(len(boxes_data)):
            if boxes_data[i] == []:
                request_copy.text_data.append(None)
                continue
            x, y, w, h = int(boxes_data[i][0]), int(boxes_data[i][1]), int(boxes_data[i][2]), int(boxes_data[i][3])
            gray = cv2.cvtColor(frame_data[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
            id_, confidence = self.recognizer.predict(gray)
            if confidence < 100:
                emotion = self.names[id_-1]
                request_copy.text_data.append(emotion)
            else:
                request_copy.text_data.append(None)

        self.expression_message_queue.put(request_copy)
