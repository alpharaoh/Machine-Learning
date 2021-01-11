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


"""
This class holds data about an object in the scene such as:
- the position
- the identifier of the object
- the centre co-ordinates
"""
class GameObject():
  
   def __init__(self, id=None, position_xywh=None):
      self.id = id
      self.position_xywh = position_xywh
      self.centre_cords = self.get_centre_of_bbox(position_xywh)

   def get_centre_of_bbox(self, bbox):
      x1, y1, x2, y2, _ = bbox

      centre_x = (x2-x1)/2 + x1
      centre_y = (y2-y1)/2 + y1

      return (centre_x, centre_y)


"""
Todo: class description
"""
class Grid():

   def __init__(self, x=1920, y=1080, ratio=(16, 9), ratio_mult=2, line_tol=1):
      self.x, self.y = x, y

      self.line_tol = line_tol

      self.ratio_x, self.ratio_y = ratio
      assert type(ratio_mult) == int, "ratio multiplier must be of type int"

      self.squares_x = self.ratio_x * ratio_mult
      self.squares_y = self.ratio_y * ratio_mult

      self.projectile_index = 1

      self.create_grid()


   def increment_projectile_index(self):
      self.projectile_index += 1

   def clear_grid(self):
      self.create_grid()
   
   def create_grid(self):

      assert (self.ratio_x / self.x)  == (self.ratio_y / self.y), f"ratio must be respective of resolution; \
         {self.ratio_x}:{self.ratio_y} does not corrispond to the resolution {self.x}x{self.y}"

      self.grid = np.zeros((self.squares_x, self.squares_y), dtype=int)

   def change_value(self, pos, value):
      x, y = pos
      self.grid[y, x] = value

   def add_xleftright(self, y, x, tol):
      change_value((x, y), self.projectile_index)

      if x > 0:
         for i in range(self.line_tol):
            grid[y, x + i + 1] = self.projectile_index
            grid[y, x - i - 1] = self.projectile_index

   def add_line(self, start_pos, gradient):
      x, y = start_pos

      # get y-intercept (c) using a rearranged version of y = mx + c
      c = y - (gradient * x)

      for y_new in range(y+1, len(grid)):
         # get x using a rearranged version of y = mx + c
         x = (y_new - c) / gradient

         # handles case when x is outside of the number of squares on the grid we have
         if abs(x) >= self.squares_x:
            break

         self.add_xleftright(y_new, int(x))

   def value_in_projectile(self, x_grid, y_grid):
      return 1 == grid[y_grid, x_grid]

   def change_square(self, pos=(400, 300)):
      pos_x, pos_y = pos

      grid_y, grid_x = res_to_grid_squares(pos_x, pos_y)

      self.grid[grid_y, grid_x] = self.projectile_index
   
   def res_to_grid_squares(self, x, y):
      rat_x, rat_y = x/self.x, y/self.y 

      grid_x = int(rat_x * self.squares_x) - 1
      grid_y = int(rat_y * self.squares_y) - 1

      return grid_x, grid_y

   def O1_2_xy(self, x, y):
      """
      This takes a normalised x, y that ranges from 0-1
      to 2 new values that range from the resolution of the
      image, e.g. from 0-1920 and 0-1080
      """

      new_x = round(x*self.x)
      new_y = round(y*self.y)

      return new_x, new_y


class previousFrameData():
   def __init__(self):
      self.frames = []

   def append(self, scene):
      self.frames.append(scene.axes)

   def show_all(self):
      for i, scenes in enumerate(self.frames):
         for j, axe in enumerate(scenes):
            print(f"Scene {i+1}: Axe {j+1} - {axe.centre_cords}")
   
   def __len__(self):
      return len(self.frames)

"""
This class holds any instances of type Object and splits them
up occordingly depending on if the object is a mundo or axe
"""
class Scene():
   
   def __init__(self, mundos=[], axes=[]):
      self.mundos = mundos
      self.axes = axes

   def __len__(self):
      return len(self.mundos) + len(self.axes)

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
      
      self.prev_dat = previousFrameData()


      while True:
         current_frame_data = next(self.gen)
         objects_in_scene = self.get_state(current_frame_data)            

         print(f"Total objects detected: {len(objects_in_scene)}")

         predictions = self.get_axe_prediction()
         ## to calcs

   
        # self.visuals.show_visuals(objects_in_scene, self.get_current_frame(), predictions)

         objects_in_scene.clear()


   def get_state(self, scene_data):  
      objects_bboxes = scene_data.xyxyn
      predictions = scene_data.pred
      
      objects_list = objects_bboxes[0]

      scene = Scene()
      scene.clear()

      axes = False

      for object_ in objects_list:
         obj = GameObject(id=int(object_[-1]), position_xywh=object_[:-1])

         if obj.id == 0: # mundo
            scene.mundos.append(obj)
         else:
            axes = True
            scene.axes.append(obj)
   
      if axes:
         self.prev_dat.append(scene)
         self.prev_dat.show_all()
         print("/\\")

      return scene

   def get_current_frame(self):
      return self.capture_data.frame


   def get_axe_prediction(self):
      if len(self.prev_dat) < self.frame_history:
         return

      self.prev_dat.show_all()

      full_data = []

      for i, scene in enumerate(self.prev_axe_data):
         centre_pos = []
         for axe_data in scene:
            centre_pos.append(axe_data.centre_cords)

         full_data.append(centre_pos)

      current_frame = full_data[2]
      after_frame = full_data[1]
      last_frame = full_data[0]

      print("Current frame:", current_frame)
      print("After frame:", after_frame)
      print("Last frame:", last_frame)

      grid = Grid(self.x, self.y)

      for pos1 in last_frame:
         for pos2 in after_frame:
            # get gradient of 2 points as a line
            grad = self.gradient((pos1, pos2))
            
            # add line to grid
            grid.add_line(pos1, grad)

            # increase index
            grid.increment_projectile_index()
      
      pred = []

      for i in current_frame:
         if grid.value_in_projectile(i):
            pred.append(i)
            

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