# emotion_and_action_detection_1
This is a test of multi-model_app.

## code logics

### Six modules: 

`VideoProcessing`: input: video;  output: frame.

`PersonRecognition`: input: frame;  output: frame, person_boxes;  
model: hustvl/yolos-tiny from https://huggingface.co/hustvl/yolos-tiny.

`FaceRecognition`: input: frame, person_boxes;  output: frame, face_boxes;  
model: cv2.CascadeClassifier('haarcascade_frontalface_default.xml') from https://github.com/omar-aymen/Emotion-recognition/tree/master.

`ExpressionRecognition`: input: frame, face_boxes;  output: frame, face_boxes, emotion;  
model: cv2.face.LBPHFaceRecognizer_create().read('trainer_emotion.yml') from https://github.com/omar-aymen/Emotion-recognition/tree/master.

`PostureRecognition`: input: frame, person_boxes; output: processed_frame;  
model: pytorch-openpose/body_pose_model.pth from https://github.com/Hzzone/pytorch-openpose

`InformationSummary`: input: frame, face_boxes, emotion, processed_frame; output: processed_video.

![Image](https://github.com/lifang535/emotion_and_action_detection_1/blob/main/app.png)

### Request in data transmission: 

1 * video : n * frame : n * m * person_box : n * m * face_box : n * m * draw_message

```
         path                         num_frame                         num_person                         num_face                         source
'input_videos/video_1.mp4'              217                               217                                121                         https://www.pexels.com/video/man-dancing-on-rooftop-2795750/
'input_videos/video_2.mp4'              324                               290                                43                          https://www.pexels.com/video/city-road-dawn-man-4734775/
'input_videos/video_3.mp4'              250                               287                                170                         https://www.pexels.com/video/man-dancing-in-front-of-a-building-with-mural-2795730/
'input_videos/video_4.mp4'              268                               270                                224                         https://www.pexels.com/video/man-dancing-hip-hop-2795745/
'input_videos/video_5.mp4'              655                               544                                522                         https://www.pexels.com/video/person-dancing-4540399/
'input_videos/video_6.mp4'              147                               153                                44                          https://www.pexels.com/video/young-man-practicing-break-dance-5363330/
'input_videos/video_7.mp4'              427                               363                                139                         https://www.pexels.com/video/man-outdoors-modern-break-dance-4734793/
'input_videos/video_8.mp4'              127                               127                                65                          https://www.pexels.com/video/city-man-dancing-sport-4734627/
'input_videos/video_9.mp4'              128                               134                                70                          https://www.pexels.com/video/emotions-dancing-amusement-park-portrait-4841885/
'input_videos/video_10.mp4'             369                               1605                               972                         https://www.pexels.com/video/a-man-doing-breakdancing-8688465/
```

### Throughout of model:

When the modules are 1 : 1 : 1 : 1 : 4 : 1 for inference, the process time of videos are:
```
PersonRecognition(1):     [..., 0.064, 0.058, 0.061, 0.061, 0.063, 0.059, 0.057, 0.061, 0.063, 0.066, 0.063, 0.062, 0.068, 0.068, 0.072, 0.069, 0.076, 0.069, 0.070, 0.076, ...]

FaceRecognition(1):       [..., 0.030, 0.032, 0.033, 0.039, 0.039, 0.041, 0.041, 0.043, 0.038, 0.034, 0.039, 0.030, 0.037, 0.037, 0.037, 0.033, 0.040, 0.042, 0.040, 0.037, ...]

ExpressionRecognition(1): [..., 0.012, 0.016, 0.013, 0.015, 0.009, 0.002, 0.017, 0.011, 0.002, 0.016, 0.017, 0.016, 0.003, 0.004, 0.003, 0.002, 0.003, 0.020, 0.018, 0.017, ...]

PostureRecognition(4):    [..., 1.177, 1.302, 1.304, 1.318, 1.304, 1.205, 1.376, 1.257, 1.341, 1.342, 1.119, 1.395, 1.341, 1.334, 1.369, 1.355, 1.318, 1.321, 1.333, 1.132, ...]
```

Throughout:
```
throughout of PersonRecognition(1) ≈ 15.31 req/s (cuda:1  MEM: 1784 MiB; UTL: 7 ~ 15％，avg-10％)

throughout of FaceRecognition(1) ≈ 26.95 req/s (cpu  UTL: 100 ~ 1000％, avg-120％)

throughout of ExpressionRecognition(1) ≈ 92.59 req/s (cpu  UTL: 10 ~ 20％, avg-12％)

throughout of PostureRecognition(4) ≈ 0.77 req/s (cuda:0  MEM: 7527 MiB; UTL: 0 ~ 10％，avg-1.5％)
```

### Output of app:

`'input_videos/video_1.mp4'` -> `'output_videos/processed_video_1.mp4'` (-> .gif)

![Video](output_videos/processed_video_1.gif)

## TODO

proportion of modules and latency calculation
