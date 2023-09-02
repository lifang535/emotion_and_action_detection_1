import os
import cv2
import time
import torch
import threading
import multiprocessing

from PIL import Image, ImageDraw
from logger import logger_information_summary

class InformationSummary(multiprocessing.Process):
    def __init__(self, id, expression_message_queue, action_message_queue, end_signal, lock):
        super().__init__()
        self.id = id
        self.expression_message_queue = expression_message_queue
        self.action_message_queue = action_message_queue
        self.end_signal = end_signal
        self.lock = lock

        self.device = None

        self.process_times = []

    def run(self):
        logger_information_summary.info(f"[InformationSummary_{self.id}] start")

        self.end_signal.value += 0

        self.device = torch.device("cuda:0")

        thread_monitor_expression = threading.Thread(target=self.monitor_expression, args=(None,))
        thread_monitor_action = threading.Thread(target=self.monitor_action, args=(None,))
        thread_monitor_expression.start()
        thread_monitor_action.start()
        thread_monitor_expression.join()
        thread_monitor_action.join()

    def monitor_expression(self, request):
        while True:
            with self.lock:
                request = self.expression_message_queue.get()
            if request.signal == -1:
                self.expression_message_queue.put(request)
                if self.expression_message_queue.qsize() != 1: # !!!
                    continue
                with self.lock:
                    logger_information_summary.info(f"[InformationSummary_{self.id}] monitor_expression get signal -1")
                self.end_signal.value -= 1
                if self.end_signal.value == 0:
                    ...
                break
            cache_path = f"cache/output_frames_{request.ids[0]}/{request.ids[1]}.jpg"
            while not os.path.exists(cache_path):
                time.sleep(0.1)
            # time.sleep(0.1)
            frame = cv2.imread(cache_path)
            for i in range(len(request.boxes_data)):
                if request.boxes_data[i] == []:
                    continue
                box_data = [int(x) for x in request.boxes_data[i]]
                cv2.rectangle(frame, (box_data[0], box_data[1]), (box_data[2], box_data[3]), color=(255, 0, 0), thickness=2)
                cv2.putText(frame, f"{request.text_data[i]}", (box_data[0], box_data[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            try:
                cv2.imwrite(cache_path, frame)
            except Exception as e:
                print(f"e: {e}")

            file_count = len(os.listdir(f"cache/output_frames_{request.ids[0]}"))
            if file_count == request.sub_requests[0] and request.ids[1] == request.sub_requests[0]:
                logger_information_summary.info(f"[InformationSummary_{self.id}] process video {request.ids[0]}")
                self.process_video(request)
            
    def monitor_action(self, request):
        while True:
            with self.lock:
                request = self.action_message_queue.get()
            if request.signal == -1:
                self.action_message_queue.put(request)
                if self.action_message_queue.qsize() != 1: # !!!
                    continue
                with self.lock:
                    logger_information_summary.info(f"[InformationSummary_{self.id}] monitor_action get signal -1")
                self.end_signal.value -= 1
                if self.end_signal.value == 0:
                    ...
                break

            cache_dir = f"cache/output_frames_{request.ids[0]}"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            cache_path = f"cache/output_frames_{request.ids[0]}/{request.ids[1]}.jpg"
            # cv2.imwrite(cache_path, request.frame_data)
            frame = Image.fromarray(request.frame_data)
            frame.save(cache_path)

    def process_video(self, request):
        cache_dir = f"cache/output_frames_{request.ids[0]}"
        output_video_dir = "output_videos"
        os.makedirs(output_video_dir, exist_ok=True)

        # Create a video from the processed frames
        output_frames = [os.path.join(cache_dir, output_frame) for output_frame in os.listdir(cache_dir)]
        output_frames.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        # print(f"output_frames: {output_frames}")
        output_video_path = f"output_videos/processed_video_{request.ids[0]}.mp4"

        # Open the video file TODO
        cap = cv2.VideoCapture(f'input_videos/video_{request.ids[0]}.mp4') # TODO

        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Write the frames to the video file
        for output_frame in output_frames:
            frame = cv2.imread(output_frame)
            out.write(frame)

        # Release the video writer
        out.release()

        # Delete the processed frames
        os.system("rm -rf " + cache_dir)
