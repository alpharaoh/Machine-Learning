# python3 detect.py --weights /Users/alpharaoh/Downloads/best\ \(1\).pt --img 416 --conf 0.6 --source ./../dataset/output/test_yolo/full_game/
from mss import mss
import cv2
import numpy as np
from PIL import Image
import sys

sys.path.append('../yolov5/')
#import detect

print("Loaded.\n")

class YoloData():
   def __init__(self, x_res, y_res, monitor_number=0, debug=False):
      self.x_res = x_res
      self.y_res = y_res

      self.debug = debug

      monitor = mss().monitors[monitor_number]

      self.settings = {
         "top": monitor["top"], 
         "left": monitor["left"], 
         "width": x_res, 
         "height": y_res, 
      }
 
   def pass_frame_to_YOLO(self, frame):
      detect.detect(save_img=False)

   def get_frame(self, sct):
      frame = sct.grab(self.settings)
      frame = Image.frombytes('RGB', (frame.size.width, frame.size.height), frame.rgb)
      frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

      return frame

   def draw_bbox_on_frame(self):
      pass
 
   def run(self):
       with mss() as sct:
        while True:
            frame = self.get_frame(sct) #numpy array

            if self.debug:
               cv2.imshow('Test', np.array(frame)) #output screen, for testing only

               if cv2.waitKey(25) & 0xFF == ord('q'): #Press Q on debug windows to exit
                  cv2.destroyAllWindows()
                  break

if __name__ == '__main__':
   yolo = YoloData(1920, 1080, monitor_number=0, debug=True)
   yolo.run()