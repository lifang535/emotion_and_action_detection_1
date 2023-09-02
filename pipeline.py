import os
import cv2
import time
import torch

import threading
import multiprocessing

from request import Request
from video_processing import VideoProcessing
from person_recognition import PersonRecognition
from face_recognition import FaceRecognition
from posture_recognition import PostureRecognition
from expression_recognition import ExpressionRecognition
from action_recognition import ActionRecognition
from information_summary import InformationSummary

if __name__ == "__main__":
    if os.path.exists("output_videos"):
        os.system("rm -r output_videos")
    os.mkdir("output_videos")
    for i in range(1, 11):
        if os.path.exists(f"cache/output_frames_{i}"):
            os.system(f"rm -r cache/output_frames_{i}")

    managers = []
    end_signals = []
    locks = []
    for i in range(8):
        managers.append(multiprocessing.Manager())
        end_signals.append(managers[i].Value("i", 0))
        # locks.append(managers[i].Lock())
        locks.append(multiprocessing.Lock())

    video_queue = managers[0].Queue()
    video_frame_queue = managers[1].Queue()
    person_box_to_face_queue = managers[2].Queue()
    person_box_to_posture_queue = managers[3].Queue()
    face_box_queue = managers[4].Queue()
    posture_message_queue = managers[5].Queue()
    expression_message_queue = managers[6].Queue()
    action_message_queue = managers[7].Queue()

    VideoProcessings = []
    PersonRecognitions = []
    FaceRecognitions = []
    PostureRecognitions = []
    ExpressionRecognitions = []
    ActionRecognitions = []
    InformationSummaries = []
    for i in range(5):
        VideoProcessings.append(VideoProcessing(i+1, video_queue, video_frame_queue, end_signals[0], locks[0]))
        PersonRecognitions.append(PersonRecognition(i+1, video_frame_queue, person_box_to_face_queue, person_box_to_posture_queue, end_signals[1], locks[1]))
        FaceRecognitions.append(FaceRecognition(i+1, person_box_to_face_queue, face_box_queue, end_signals[2], locks[2]))
        PostureRecognitions.append(PostureRecognition(i+1, person_box_to_posture_queue, posture_message_queue, end_signals[3], locks[3]))
        ExpressionRecognitions.append(ExpressionRecognition(i+1, face_box_queue, expression_message_queue, end_signals[4], locks[4]))
        ActionRecognitions.append(ActionRecognition(i+1, posture_message_queue, action_message_queue, end_signals[5], locks[5]))
        InformationSummaries.append(InformationSummary(i+1, expression_message_queue, action_message_queue, end_signals[6], locks[6]))

    num_models = [1, 1, 4, 4, 4, 4, 1]

    for i in range(num_models[0]):
        VideoProcessings[i].start()
    for i in range(num_models[1]):
        PersonRecognitions[i].start()
    for i in range(num_models[2]):
        FaceRecognitions[i].start()
    for i in range(num_models[3]):
        PostureRecognitions[i].start()
    for i in range(num_models[4]):
        ExpressionRecognitions[i].start()
    for i in range(num_models[5]):
        ActionRecognitions[i].start()
    for i in range(num_models[6]):
        InformationSummaries[i].start()

    input_video_dir = "input_videos"
    input_video_files = os.listdir(input_video_dir)
    input_video_files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    time.sleep(5) # wait for model initialization
    # input_video_files.pop(-1) # !!!
    for input_video_file in input_video_files:
        time.sleep(10)
        video_id = int(input_video_file.split(".")[0].split("_")[-1])
        request = Request(ids=[video_id],
                            sub_requests=[],
                            video_data=input_video_dir + "/" + input_video_file,
                            frame_data=None,
                            boxes_data=None,
                            text_data=None,
                            signal=None,
                            start_time=time.time())
        video_queue.put(request)

    # put end signal into queue
    request = Request(ids=[],
                        sub_requests=[],
                        video_data=None,
                        frame_data=None,
                        boxes_data=None,
                        text_data=None,
                        signal=-1,
                        start_time=time.time())
    video_queue.put(request)

    for i in range(num_models[0]):
        VideoProcessings[i].join()
    for i in range(num_models[1]):
        PersonRecognitions[i].join()
    for i in range(num_models[2]):
        FaceRecognitions[i].join()
    for i in range(num_models[3]):
        PostureRecognitions[i].join()
    for i in range(num_models[4]):
        ExpressionRecognitions[i].join()
    for i in range(num_models[5]):
        ActionRecognitions[i].join()
    for i in range(num_models[6]):
        InformationSummaries[i].join()
