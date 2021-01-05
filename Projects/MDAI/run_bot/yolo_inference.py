"""
This program will turn the frames from a screen grab of a display
and do YOLO inference on this frame to return the data of the 
current frame
"""

import torch
from PIL import Image
import torch
import cv2
from mss import mss

from screen_capture import ScreenCapture

"""
pytorch Detectors object class:
'__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', 
'__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 
'__weakref__', 'display', 'imgs', 'n', 'names', 'pred', 'print', 'save', 'show', 'tolist', 'xywh', 'xywhn', 'xyxy', 'xyxyn']
"""

class YoloInference():
   def __init__(self, weights, conf=0.6, draw_boxes=True):
      self.draw_boxes = draw_boxes
      self.screen_capture = ScreenCapture(1920, 1080, monitor_number=1)

      # load our model (doesn't need yolo repo)
      self.model = model = torch.hub.load("ultralytics/yolov5", 
                                          "custom", 
                                          path_or_model=weights)

   def infer_real_time_frames(self):
      with mss() as sct:
         while True:
            # get frame from screen grab
            frame = self.screen_capture.get_frame(sct)

            # use YOLO model to infer frame
            detection = self.model(frame, size=416)

            ##### todo: Calculate projectile of axe frame frames 

            yield detection.xyxy
   

YoloInference("/Users/alpharaoh/Downloads/best (1).pt").bar()