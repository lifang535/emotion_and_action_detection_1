import os
import cv2
import time
import torch
import threading
import multiprocessing
import numpy as np

from PIL import Image
from logger import logger_person_recognition
from transformers import YolosImageProcessor, YolosForObjectDetection

class PersonRecognition(multiprocessing.Process):
    def __init__(self, id, video_frame_queue, person_box_to_face_queue, person_box_to_posture_queue, end_signal, lock):
        super().__init__()
        self.id = id
        self.video_frame_queue = video_frame_queue
        self.person_box_to_face_queue = person_box_to_face_queue
        self.person_box_to_posture_queue = person_box_to_posture_queue
        self.end_signal = end_signal
        self.lock = lock

        self.device = None
        self.model = None
        self.processor = None

        self.process_times = []

    def run(self):
        logger_person_recognition.info(f"[PersonRecognition_{self.id}] start")

        self.end_signal.value += 1

        self.device = torch.device("cuda:0")
        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').to(self.device)
        self.processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

        while True:
            with self.lock:
                request = self.video_frame_queue.get()
            if request.signal == -1:
                self.video_frame_queue.put(request)
                with self.lock:
                    logger_person_recognition.info(f"[PersonRecognition_{self.id}] get signal -1")
                self.end_signal.value -= 1
                if self.end_signal.value == 0:
                    self.person_box_to_face_queue.put(request)
                    self.person_box_to_posture_queue.put(request)
                break
            process_start_time = time.time()
            self.process_frame(request)
            process_end_time = time.time()
            self.process_times.append(round(process_end_time - process_start_time, 3))

        # with self.lock:
            # logger_person_recognition.info(f"[PersonRecognition_{self.id}] process times: {self.process_times}")

    def process_frame(self, request):
        video_id, frame_id, frame_data = request.ids[0], request.ids[1], request.frame_data

        frame = Image.fromarray(frame_data)
        inputs = self.processor(frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([frame.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        person_boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if self.model.config.id2label[label.item()] == "person":
                person_boxes.append(box.tolist())
        
        request_copy_1 = request.copy()
        request_copy_1.sub_requests.append(len(person_boxes)) # to check if the box is person-box
        request_copy_1.boxes_data = person_boxes
        self.person_box_to_face_queue.put(request_copy_1)

        request_copy_2 = request.copy()
        request_copy_2.sub_requests.append(len(person_boxes)) # to check if the box is person-box
        request_copy_2.boxes_data = person_boxes
        self.person_box_to_posture_queue.put(request_copy_2)

        # request_copy_2 = request.copy()
        # request_copy_2.sub_requests.append(len(person_boxes))
        # for box in person_boxes:
        #     box = [int(i) for i in box]
        #     frame = frame.crop(box)
        #     request_copy_2.frame_data = np.array(frame)
        #     # box = [int(i) for i in box]
        #     # request_copy_2.frame_data = frame_data[box[1]:box[3], box[0]:box[2]]
        #     self.person_box_to_face_queue.put(request_copy_2)

        # if len(person_boxes) == 0: # TODO: put all boxes with the frame into queue 
        #     self.person_box_to_face_queue.put(request)
        #     self.person_box_to_posture_queue.put(request)
        #     return
