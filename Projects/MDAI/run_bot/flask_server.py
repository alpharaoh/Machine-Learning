from flask import Flask, render_template, Response
import cv2
from mss import mss

import screen_capture

app = Flask(__name__)

@app.route("/")
def index():
   """Video streaming home page."""
   return render_template("index.html")

def gen(screen_capture):
   with mss() as sct:
      while True:
         frame = screen_capture.get_frame(sct)
         frame = cv2.imencode(".jpg", frame)[1].tobytes()

         yield (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

@app.route("/video_feed")
def video_feed():
   return Response(gen(screen_capture.ScreenCapture(1920, 1080, monitor_number=1)), 
                       mimetype="multipart/x-mixed-replace; boundary=frame")
    