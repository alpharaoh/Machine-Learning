"""
This program will be used as the main brains of the bot.
To calculate an axe projectiles path based on previous frames
or the position of the enemy mundo is an example of what this 
program should do.
"""

"""
pred:
tensor([[1.07240e+03, 4.19211e+02, 1.21288e+03, 5.22414e+02, 8.10836e-01, 0.00000e+00],
        [7.86444e+02, 2.42541e+02, 9.04286e+02, 3.63986e+02, 7.41388e-01, 0.00000e+00],
        [1.16223e+03, 4.47822e+02, 1.19791e+03, 5.17082e+02, 4.22028e-01, 1.00000e+00],
        [8.80947e+02, 3.00264e+02, 9.54389e+02, 3.63703e+02, 3.99887e-01, 1.00000e+00]])]
"""
import time
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import cv2
import time
import random
import numpy as np
import os
from collections import deque

from yolo_inference import YoloInference
from visualisation import Visualisation

os.system("clear")
print("Loaded modules.\n\nStarting...")


NAMES = ["Mundo", "Axe"]



class GameObject():
   """
   This class holds data about an object in the scene such as:
   - the position
   - the identifier of the object
   - the centre co-ordinates
   """
   def __init__(self, id=None, position_xywh=None):
      self.id = id
      self.position_xywh = position_xywh
      self.centre_cords = self.get_centre_of_bbox(position_xywh)

   def get_centre_of_bbox(self, bbox):
      x1, y1, x2, y2, _ = bbox

      centre_x = (x2-x1)/2 + x1
      centre_y = (y2-y1)/2 + y1

      return (centre_x, centre_y)

class Grid():
   def __init__(self, x=1920, y=1080, ratio=(16, 9), ratio_mult=2):
      # y = mx + c
      self.x, self.y = x, y

      self.ratio_x, self.ratio_y = ratio
      assert type(ratio_mult) == int, "ratio multiplier must be of type int"

      self.squares_x = self.ratio_x * self.ratio_mult
      self.squares_y = self.ratio_y * self.ratio_mult

      self.ratio_mult = ratio_mult

      self.split_grid()
   
   def split_grid(self):

      assert (self.ratio_x / self.x)  == (self.ratio_y / self.y), f"ratio must be respective of resolution; \
         {self.ratio_x}:{self.ratio_y} does not corrispond to the resolution {self.x}x{self.y}"

      self.grid = np.zeros((self.squares_x, self.squares_y), dtype=int)

   def highlight_square(self, pos=(400, 300)):
      pos_x, pos_y = pos

      square_x, square_y = pos_x/self.x, pos_y/self.y 

      grid_x = int(square_x * self.squares_x) - 1
      grid_y = int(square_y * self.squares_y) - 1

      self.grid[grid_y, grid_x] = 1
      

class Scene():
   """
   This class holds any instances of type Object and splits them
   up occordingly depending on if the object is a mundo or axe
   """
   def __init__(self, mundos=[], axes=[]):
      self.mundos = mundos
      self.axes = axes

   def __len__(self):
      return len(self.mundos + self.axes)

   def clear(self):
      self.mundos.clear()
      self.axes.clear()


class GameAnalysis():
   
   def __init__(self, weights=".", frame_history=3):
      assert type(weights) == str, "Weights are path of .pt weights file as Type String"
      self.capture_data = YoloInference(weights)
      self.gen = self.capture_data.infer_real_time_frames()

      self.x = self.capture_data.screen_capture.x_res
      self.y = self.capture_data.screen_capture.y_res

      self.visuals = Visualisation(self.x, self.y)
      self.frame_history = frame_history
      
      self.prev_axe_data = deque([])


      while True:
         current_frame_data = next(self.gen)
         objects_in_scene = self.get_state(current_frame_data)            

         print(f"Total objects detected: {len(objects_in_scene)}")

         predictions = self.get_axe_prediction()
         ## to calcs

   
         self.visuals.show_visuals(objects_in_scene, self.get_current_frame(), predictions)

         objects_in_scene.clear()


   def get_state(self, scene_data):  
      objects_bboxes = scene_data.xyxyn
      predictions = scene_data.pred
      
      objects_list = objects_bboxes[0]

      scene = Scene()

      axes = False

      for object_ in objects_list:
         obj = GameObject(id=int(object_[-1]), position_xywh=object_[:-1])

         if obj.id == 0: # mundo
            scene.mundos.append(obj)
         else:
            axes = True
            scene.axes.append(obj)
   
      if axes:
         self.add_to_axe_frames(scene)

      return scene

   def get_current_frame(self):
      return self.capture_data.frame

   def add_to_axe_frames(self, frame_data):

      if len(self.prev_axe_data) == self.frame_history:
         self.prev_axe_data.rotate(-1)
         self.prev_axe_data.pop()
         self.prev_axe_data.append(frame_data)   

      else:
         self.prev_axe_data.append(frame_data)


   def get_axe_prediction(self):
      if len(self.prev_axe_data) < self.frame_history:
         return

      full_data = []

      for i, scene in enumerate(self.prev_axe_data):
         centre_pos = []
         for axe_data in scene.axes:
            centre_pos.append(axe_data.centre_cords)

         full_data.append(centre_pos)

      print(full_data)

      current_frame = full_data[2]
      after_curr_frame = full_data[1]
      last_frame = full_data[0]

      grid = Grid(self,x, self.y)

      for point in current_frame:
         pass


      return pred

   def parallel(self, line1, line2):
      if near(gradient(line1), gradient(line2)):
         return gradient(line1), line2

   def gradient(self, line):
      (x1, y1), (x2, y2) = line

      # Ensure that the line is not vertical
      if x1 != x2:
         gradient = (1. / (x1 - x2)) * (y1 - y2)
         return gradient   

def near(self, val1, val2, rtol=1e-2, atol=1e-3):
      return np.isclose(val1, val2, rtol=rtol, atol=atol)

game = GameAnalysis(weights="/Users/alpharaoh/Downloads/best (1).pt")