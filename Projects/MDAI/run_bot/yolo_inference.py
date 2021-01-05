import torch
from PIL import Image
import torch
import cv2
from mss import mss

from screen_capture import ScreenCapture

"""
'__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', 
'__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 
'__weakref__', 'display', 'imgs', 'n', 'names', 'pred', 'print', 'save', 'show', 'tolist', 'xywh', 'xywhn', 'xyxy', 'xyxyn']
"""

class MundoAIFramework():
   def __init__(self, weights, conf=0.6, draw_boxes=True):
      self.draw_boxes = draw_boxes
      self.screen_capture = ScreenCapture(1920, 1080, monitor_number=1)
      self.model = model = torch.hub.load("ultralytics/yolov5", 
                                          "custom", 
                                          path_or_model=weights)

   def bar(self):
      with mss() as sct:
         while True:
            frame = self.screen_capture.get_frame(sct)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            detection = self.model(frame, size=416)

            print(detection.xyxy)
   

MundoAIFramework("/Users/alpharaoh/Downloads/best (1).pt").bar()