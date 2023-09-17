# watchouse

Python Application to monitor streaming video to detect people and trigger alerts.

Initially custom to use in a Raspberry Pi and support only RTSP as a Streaming Source.

The script is main.py and the envs is required:
- HOST
- PORT
- USR
- PASSWORD

Personalize the check_trigger function to make request to your favorite service trigger a push. (My is Push-bullet)

# Dependencies
- OpenCV (https://opencv.org/)
- YoloV5 Model (https://github.com/ultralytics/yolov5)
- torch
- torchvision

The model used is the smallest model: models/yolov5n.pt