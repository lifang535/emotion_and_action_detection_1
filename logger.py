import logging

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

logger_video_processing = logging.getLogger('logger_video_processing')
logger_video_processing.setLevel(logging.INFO)
fh_video_processing = logging.FileHandler('logs/video_processing.log', mode='w')
fh_video_processing.setLevel(logging.INFO)
fh_video_processing.setFormatter(formatter)
logger_video_processing.addHandler(fh_video_processing)

logger_person_recognition = logging.getLogger('logger_person_recognition')
logger_person_recognition.setLevel(logging.INFO)
fh_person_recognition = logging.FileHandler('logs/person_recognition.log', mode='w')
fh_person_recognition.setLevel(logging.INFO)
fh_person_recognition.setFormatter(formatter)
logger_person_recognition.addHandler(fh_person_recognition)

logger_face_recognition = logging.getLogger('logger_face_recognition')
logger_face_recognition.setLevel(logging.INFO)
fh_face_recognition = logging.FileHandler('logs/face_recognition.log', mode='w')
fh_face_recognition.setLevel(logging.INFO)
fh_face_recognition.setFormatter(formatter)
logger_face_recognition.addHandler(fh_face_recognition)

logger_posture_recognition = logging.getLogger('logger_posture_recognition')
logger_posture_recognition.setLevel(logging.INFO)
fh_posture_recognition = logging.FileHandler('logs/posture_recognition.log', mode='w')
fh_posture_recognition.setLevel(logging.INFO)
fh_posture_recognition.setFormatter(formatter)
logger_posture_recognition.addHandler(fh_posture_recognition)

logger_expression_recognition = logging.getLogger('logger_expression_recognition')
logger_expression_recognition.setLevel(logging.INFO)
fh_expression_recognition = logging.FileHandler('logs/expression_recognition.log', mode='w')
fh_expression_recognition.setLevel(logging.INFO)
fh_expression_recognition.setFormatter(formatter)
logger_expression_recognition.addHandler(fh_expression_recognition)

logger_action_recognition = logging.getLogger('logger_action_recognition')
logger_action_recognition.setLevel(logging.INFO)
fh_action_recognition = logging.FileHandler('logs/action_recognition.log', mode='w')
fh_action_recognition.setLevel(logging.INFO)
fh_action_recognition.setFormatter(formatter)
logger_action_recognition.addHandler(fh_action_recognition)

logger_information_summary = logging.getLogger('logger_information_summary')
logger_information_summary.setLevel(logging.INFO)
fh_information_summary = logging.FileHandler('logs/information_summary.log', mode='w')
fh_information_summary.setLevel(logging.INFO)
fh_information_summary.setFormatter(formatter)
logger_information_summary.addHandler(fh_information_summary)


