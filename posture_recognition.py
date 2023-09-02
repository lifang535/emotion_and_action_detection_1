import os
import cv2
import time
import torch
import threading
import multiprocessing
import numpy as np

import copy
from PIL import Image
from logger import logger_posture_recognition
from src import model
from src import util
from src.body import Body
from src.hand import Hand

class PostureRecognition(multiprocessing.Process):
    def __init__(self, id, person_box_to_posture_queue, posture_message_queue, end_signal, lock):
        super().__init__()
        self.id = id
        self.person_box_to_posture_queue = person_box_to_posture_queue
        self.posture_message_queue = posture_message_queue
        self.end_signal = end_signal
        self.lock = lock

        self.device = None
        self.body_estimation = None

        self.process_times = []

    def run(self):
        logger_posture_recognition.info(f"[PostureRecognition_{self.id}] start")

        self.end_signal.value += 1

        self.device = torch.device("cuda:0")
        self.body_estimation = Body('pytorch-openpose/body_pose_model.pth')

        while True:
            with self.lock:
                request = self.person_box_to_posture_queue.get()
            if request.signal == -1:
                self.person_box_to_posture_queue.put(request)
                with self.lock:
                    logger_posture_recognition.info(f"[PostureRecognition_{self.id}] get signal -1")
                self.end_signal.value -= 1
                if self.end_signal.value == 0:
                    self.posture_message_queue.put(request)
                break
            process_start_time = time.time()
            self.process_person_box(request)
            process_end_time = time.time()
            self.process_times.append(round(process_end_time - process_start_time, 3))

        # with self.lock:
            # logger_posture_recognition.info(f"[PostureRecognition_{self.id}] process times: {self.process_times}")

    def process_person_box(self, request):
        video_id, frame_id, frame_data, boxes_data = request.ids[0], request.ids[1], request.frame_data, request.boxes_data

        cache_path = f"cache/frame_{video_id}-{frame_id}.jpg"
        frame = Image.fromarray(frame_data).save(cache_path) # RGB
        frame = cv2.imread(cache_path) # BGR
        candidate, subset = self.body_estimation(frame)
        frame = util.draw_bodypose(frame, candidate, subset)

        request_copy = request.copy()
        request_copy.frame_data = np.array(frame)[..., ::-1] # BGR -> RGB

        self.posture_message_queue.put(request_copy)

        # print(f"[PostureRecognition_{self.id}] request {request_copy.ids}")

        os.system("rm -rf " + cache_path)
