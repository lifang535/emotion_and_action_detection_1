import os
import cv2
import time
import torch
import threading
import multiprocessing

from logger import logger_action_recognition

class ActionRecognition(multiprocessing.Process):
    def __init__(self, id, posture_message_queue, action_message_queue, end_signal, lock):
        super().__init__()
        self.id = id
        self.posture_message_queue = posture_message_queue
        self.action_message_queue = action_message_queue
        self.end_signal = end_signal
        self.lock = lock

        self.device = None

        self.process_times = []

    def run(self):
        logger_action_recognition.info(f"[ActionRecognition_{self.id}] start")

        self.end_signal.value += 1

        self.device = torch.device("cuda:0")

        while True:
            with self.lock:
                request = self.posture_message_queue.get()
            if request.signal == -1:
                self.posture_message_queue.put(request)
                with self.lock:
                    logger_action_recognition.info(f"[ActionRecognition_{self.id}] get signal -1")
                self.end_signal.value -= 1
                if self.end_signal.value == 0:
                    self.action_message_queue.put(request)
                break
            process_start_time = time.time()
            self.process_posture(request)
            process_end_time = time.time()
            self.process_times.append(round(process_end_time - process_start_time, 3))

        # with self.lock:
            # logger_action_recognition.info(f"[ActionRecognition_{self.id}] process times: {self.process_times}")

    def process_posture(self, request): # TODO: action recognition
        request_copy = request.copy()
        self.action_message_queue.put(request_copy)
