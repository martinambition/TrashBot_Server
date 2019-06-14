import cv2
import sys
import os
from multiprocessing import Process, Queue
import threading
from trash_model import TrashModel
from PIL import Image
import torch
import numpy as np
import cv2
from torchvision.transforms import functional as F
from web_socket import SocketServer

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def run():
    print("main process:",os.getpid())

    q = Queue()

    # start face recognization in child process
    socket_server =  SocketServer(50010,q)
    fw = Process(target=socket_server)
    fw.start()


    num_classes = 3  # Background, Shoe, Cola
    classes = {"Shoes": 1, "Cola": 2}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = TrashModel(num_classes)
    model.load_state_dict(torch.load("trash.pt"))
    model.eval()

    while (True):
        image_data = q.get(True)
        nparr = np.frombuffer(image_data, np.uint8)

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        frame = rotateImage(frame,90)
        rec_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #
        #rec_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # path = "/Users/i303138/Documents/Learning/MachineLearning/Projects/Lego/Dataset/JPEGImages/Cola/IMG_20190607_105703.jpg"

        img = F.to_tensor(rec_img)
        pre = model([img])
        height = frame.shape[0]
        ratio = 500 / height

        frame = cv2.resize(frame, (int(ratio * frame.shape[1]), int(ratio * frame.shape[0])))

        if len(pre) > 0:
            scores = pre[0]['scores'].data.tolist()
            if len(scores) > 0 and scores[0] > 0.5:
                rect = pre[0]['boxes'][0].data.numpy()
                scale_rect = (rect * ratio).astype(int)
                cv2.rectangle(frame, (scale_rect[0], scale_rect[1]), (scale_rect[2], scale_rect[3]), (255, 0, 0), 2)
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xFF

if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))
    run()