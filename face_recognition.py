import os
import cv2
import time
import torch
import threading
import multiprocessing
import numpy as np

from PIL import Image
from logger import logger_face_recognition

class FaceRecognition(multiprocessing.Process):
    def __init__(self, id, person_box_to_face_queue, face_box_queue, end_signal, lock):
        super().__init__()
        self.id = id
        self.person_box_to_face_queue = person_box_to_face_queue
        self.face_box_queue = face_box_queue
        self.end_signal = end_signal
        self.lock = lock

        self.device = None
        self.face_detector = None

        self.process_times = []

    def run(self):
        logger_face_recognition.info(f"[FaceRecognition_{self.id}] start")

        self.end_signal.value += 1

        self.device = torch.device("cuda:0")
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        while True:
            with self.lock:
                request = self.person_box_to_face_queue.get()
            if request.signal == -1:
                self.person_box_to_face_queue.put(request)
                with self.lock:
                    logger_face_recognition.info(f"[FaceRecognition_{self.id}] get signal -1")
                self.end_signal.value -= 1
                if self.end_signal.value == 0:
                    self.face_box_queue.put(request)
                break
            process_start_time = time.time()
            self.process_person_box(request)
            process_end_time = time.time()
            self.process_times.append(round(process_end_time - process_start_time, 3))

        # with self.lock:
            # logger_face_recognition.info(f"[FaceRecognition_{self.id}] process times: {self.process_times}")

    def process_person_box(self, request):
        video_id, frame_id, frame_data, boxes_data = request.ids[0], request.ids[1], request.frame_data, request.boxes_data

        request_copy = request.copy()

        for i in range(len(boxes_data)):
            x, y, w, h = int(boxes_data[i][0]), int(boxes_data[i][1]), int(boxes_data[i][2]), int(boxes_data[i][3])
            frame = Image.fromarray(cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB))
            person_data = np.array(frame.crop((x, y, x + w, y + h)))
            gray_frame = cv2.cvtColor(person_data[..., ::-1], cv2.COLOR_BGR2GRAY)
            face_boxes = self.face_detector.detectMultiScale(gray_frame) # These are x, y, w, h
            # print(f"boxes_data[i]: {boxes_data[i]}]")
            # print(f"face_boxes: {face_boxes}")
            # logger_face_recognition.info(f"[FaceRecognition_{self.id}] boxes_data[i]: {boxes_data[i]}")
            # logger_face_recognition.info(f"[FaceRecognition_{self.id}] face_boxes: {face_boxes}")
            if len(face_boxes) > 0:
                face_boxes = [face_boxes[np.argmax(face_boxes[:, 3])]]
                # convert to x1, y1, x2, y2
                face_boxes[0][2] += face_boxes[0][0]
                face_boxes[0][3] += face_boxes[0][1]
                request_copy.boxes_data[i] = face_boxes[0] + [boxes_data[i][0], boxes_data[i][1], boxes_data[i][0], boxes_data[i][1]] # TODO: 有几帧偏差很大
            else:
                request_copy.boxes_data[i] = []

        self.face_box_queue.put(request_copy)
