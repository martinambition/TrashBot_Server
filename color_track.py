import cv2
import numpy as np



class ColorTrack():
    def __init__(self):
        pass

    def detect_green(self,frame):
        return self.detect_color(frame,np.array([33,80,40]),np.array([102, 255, 255]))
    def detect_red(self,frame):
        return self.detect_color(frame,np.array([78, 43, 46]), np.array([99, 255, 255]))

    def detect_color(self,frame,lower_bound,uper_bound):
        imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, lower_bound, uper_bound)
        kernelOpen = np.ones((5, 5))
        kernelClose = np.ones((20, 20))

        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
        maskFinal = maskClose
        conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgHSV, conts, -1, (255, 0, 0), 3)

        max_x = 0
        max_y = 0
        max_w = 0
        max_h = 0
        max_area = 0
        for i in range(len(conts)):
            x, y, w, h = cv2.boundingRect(conts[i])
            if w * h > max_area:
                max_x = x
                max_y = y
                max_w = w
                max_h = h
                max_area = w * h

        return max_x, max_y, max_w, max_h,max_area


