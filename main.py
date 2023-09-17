import os
import platform
import time
import sys

import cv2
import torch

from pathlib import Path
import numpy as np
from ultralytics.utils.plotting import Annotator, save_one_box, colors
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (check_img_size, Profile,
                           increment_path, scale_boxes,
                           non_max_suppression)

from utils.torch_utils import select_device


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Set envs
DVC = 'cpu'  # or 0, 1=GPU
WEIGHTS = 'models/yolov5n.pt'  # or 'yolov5s'
IMGSZ = (640, 640)
max_det = 10  # max objects detections
view_img = True  # check_imshow(warn=True)
vid_stride = 1
bs = 1
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
classes = [0, 15, 16]  # 0=person, 15=cat, 16=dog or None to use all classes
agnostic_nms = False
line_thickness = 3  # bounding box thickness (pixels)
ds_mode = "stream"

# Directories
save_txt = False
save_crop = False  # save cropped prediction boxes
project = ROOT / 'temp'
save_dir = increment_path(Path(project) / 'exp', exist_ok=False)
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Load model
device = select_device(DVC)
model = DetectMultiBackend(WEIGHTS, device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(IMGSZ, s=stride)  # check image size

# Run inference
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

# Load Stream

# Enable this env if you streaming use UDP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

USR = os.getenv("USR", "admin")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT", "554")

if not (PASSWORD or HOST):
    print("required envs HOST or PASSWORD not found.")
    exit(1)

source = f"rtsp://{USR}:{PASSWORD}@{HOST}:{PORT}/onvif2"
path = [source]

vcap = cv2.VideoCapture(source)
if not vcap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vcap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
frames = max(int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback


def check_trigger(_frame, detected_classes):
    if detected_classes and int(detected_classes.get('person') or 0) > 0:
        # make http request send push and frame -> _frame
        return True


def process_predictions(_pred, im0s, _seen, s):
    for i, det in enumerate(_pred):  # per image
        _seen += 1
        p, _im0, _frame = path[i], im0s[i].copy(), 1
        s += f'{i}: '

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if ds_mode == 'image' else f'_{_frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(_im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = _im0.copy() if save_crop else _im0  # for save_crop
        annotator = Annotator(_im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], _im0.shape).round()

            # Print results
            detected_classes = {}
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                detected_classes[names[int(c)]] = int(n)
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            print(detected_classes)

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                # with open(f'{txt_path}.txt', 'a') as f:
                #    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    hide_labels = False
                    hide_conf = hide_labels
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            check_trigger(annotator.result(), detected_classes)

        # Stream results
        _im0 = annotator.result()
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(str(p), _im0.shape[1], _im0.shape[0])
            cv2.imshow(str(p), _im0)
            cv2.waitKey(1)  # 1 millisecond


seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
count = 0
_s = ""
while True:
    vcap.grab()
    ret, frame = vcap.retrieve()  # vcap.read()
    if not ret:
        time.sleep(0.01)

        if not vcap.isOpened():
            print('WARNING: Reopen Video stream.')
            vcap.open(source)
        continue

    im0 = [frame.copy()]
    im = np.stack([letterbox(x, imgsz, stride=stride, auto=pt)[0] for x in im0])  # resize
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)  # contiguous

    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    process_predictions(pred, im0, seen, _s)

    count = count + 1
    cv2.waitKey(1)


vcap.release()
cv2.destroyAllWindows()
