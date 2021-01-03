"""
This program will grab the users screen and pass the frame to YOLO for it to be able
to infer where the mundo(s) or axe(s) are. 

Todo: 

- Give frame data to YOLO and get bounding boxes of objects

Currently trying a method to send frame data to Flask server and run the YOLO detect.py using http
connection
"""

import cv2
from flask import Flask, render_template, Response
from mss import mss
import numpy as np
from PIL import Image
import sys

app = Flask(__name__)

@app.route("/")
def index():
   """Video streaming home page."""
   return render_template("index.html")

def gen(image):
   print("d")
   frame = cv2.imencode(".jpg", image)[1].tobytes()
   yield (b"--frame\r\n"b"Content-Type: o,age/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed(image):
   return Response(gen(image), mimetype="multipart/x-mixed-replace; boundary=frame")


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
 
   def frame_to_server(self, frame):
      video_feed(frame)

   def get_frame(self, sct):
      # get frame from screen
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
            self.frame_to_server(frame)

            # if self.debug:
            #    cv2.imshow('Test', np.array(frame)) #output screen, for testing only

            #    if cv2.waitKey(25) & 0xFF == ord('q'): #Press Q on debug windows to exit
            #       cv2.destroyAllWindows()
            #       break

if __name__ == '__main__':
   yolo = YoloData(1920, 1080, monitor_number=1, debug=True)
   yolo.run()
