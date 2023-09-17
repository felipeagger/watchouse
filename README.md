# watchouse

Python Application to monitor streaming video to detect people and trigger alerts.

Initially custom to use in a Raspberry Pi and support only RTSP as a Streaming Source.

The script is main.py and the envs is required:
- HOST
- PORT
- USR
- PASSWORD

Personalize the send_alert function to make request to your favorite service trigger a push. (My is Push-bullet)
And in this case the ENV ACCESS_TOKEN is required.

# Dependencies
- Python >= 3.9
- OpenCV (https://opencv.org/)
- YoloV5 Model (https://github.com/ultralytics/yolov5)
- torch
- torchvision
- memoization (to make a memory cache and notify in 5 minute windows)

The model used is the smallest model: models/yolov5n.pt