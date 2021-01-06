"""
This program will be used as the main brains of the bot.
To calculate an axe projectiles path based on previous frames
or the position of the enemy mundo is an example of what this 
program should do.
"""
import time
import tensorflow as tf
import numpy as np
from yolo_inference import YoloInference

class GameAnalysis():
   def __init__(self):
      self.capture_data = YoloInference("/Users/alpharaoh/Downloads/best (1).pt")
      self.gen = self.capture_data.infer_real_time_frames()

      while True:
         current_frame_data = next(self.gen)
         mundo_data, axe_data = self.get_state(current_frame_data)

   def get_state(self, scene_data):
      objects_bboxes = scene_data.xyxyn
      return 1, 1


GameAnalysis()