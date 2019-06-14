from trash_model import TrashModel
from PIL import Image
import torch
import numpy as np
import cv2
from torchvision.transforms import functional as F

num_classes = 3 #Background, Shoe, Cola
classes = {"Shoes":1,"Cola":2}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = TrashModel(num_classes)
model.load_state_dict(torch.load("trash.pt"))
model.eval()

video_capture = cv2.VideoCapture(0)
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

while(True):
    ret, frame = video_capture.read()
    rec_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #path = "/Users/i303138/Documents/Learning/MachineLearning/Projects/Lego/Dataset/JPEGImages/Cola/IMG_20190607_105703.jpg"

    img = F.to_tensor(rec_img)

    pre = model([img])
    height = frame.shape[0]
    ratio = 500 / height

    frame = cv2.resize(frame, (int(ratio * frame.shape[1]), int(ratio * frame.shape[0])))

    if len(pre)>0:
        scores = pre[0]['scores'].data.tolist()
        if len(scores)>0 and scores[0] > 0.5:
            rect = pre[0]['boxes'][0].data.numpy()
            scale_rect = (rect * ratio).astype(int)
            cv2.rectangle(frame, (scale_rect[0], scale_rect[1]), (scale_rect[2], scale_rect[3]), (255, 0, 0),2)
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF