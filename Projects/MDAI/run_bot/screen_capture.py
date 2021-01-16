"""
This program will grab frames from a user display
"""
import cv2
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


# Test:
# sc = ScreenCapture(1920, 1080)

# with mss() as sct:
#    while True:
#       image = sc.get_frame(sct)
#       cv2.imshow("visualisation", image)

#       if cv2.waitKey(25) & 0xFF == ord('q'):
#          cv2.destroyAllWindows()
#          exit()