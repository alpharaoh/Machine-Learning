"""
This program will grab the users screen and pass the frame to YOLO for it to be able
to infer where the mundo(s) or axe(s) are. 

Todo: 

- Give frame data to YOLO and get bounding boxes of objects

Currently trying a method to send frame data to Flask server and run the YOLO detect.py using http
connection
"""
import cv2
"""
This program will grab frames from a user display
"""

from mss import mss
import numpy as np
from PIL import Image

class ScreenCapture():
   def __init__(self, x_res, y_res, monitor_number=0):
      self.x_res = x_res
      self.y_res = y_res

      # get monitor xy
      monitor = mss().monitors[monitor_number]

      self.settings = {
         "top": monitor["top"], 
         "left": monitor["left"], 
         "width": x_res, 
         "height": y_res, 
      }

   def get_frame(self, sct):
      # get frame from screen
      frame = sct.grab(self.settings)

      # convert BGR to RGB for inference
      frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)

      return frame