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
from data_struct import *

# os.system("clear")
print("Loaded modules.\n\nStarting...")


NAMES = ["Mundo", "Axe"]

"""
This class will run the screen capture and use the YOLO inference to
calculate the game state and what action to take based on the state
"""
class GameAnalysis():
   
   def __init__(self, weights=".", frame_history=3):
      assert type(weights) == str, "Weights are path of .pt weights file as Type String"
      self.capture_data = YoloInference(weights)

      # get screen capture frames as generator
      self.gen = self.capture_data.infer_real_time_frames()

      # get screen resolution in pixels e.g. 1920 x 1080
      self.x = self.capture_data.screen_capture.x_res
      self.y = self.capture_data.screen_capture.y_res

      self.visuals = Visualisation(self.x, self.y)

      # integer type that specifies how many previous frames the program will store 
      self.frame_history = frame_history
      
      self.prev_dat = previousFrameData()
      
      # loop to allow calling generator
      while True:
         current_frame_data = next(self.gen)

         # get inferance from frame
         objects_in_scene = self.get_state(current_frame_data)            

         print(".")

         # get prediction of axe projection using previous frames
         predictions = self.get_axe_prediction()
   
         self.visuals.show_visuals(objects_in_scene, self.get_current_frame(), predictions)

         objects_in_scene.clear()


   def get_state(self, scene_data):
      """
      This function will convert the Tensor data as a scene object that contains
      the mundos' data and the axes' data.
      If any axes are present within the scene, it will store these and use it
      to predict axe projectile
      """
      # get xyxy bounding box
      objects_bboxes = scene_data.xyxyn

      # convert object bounding box Tensor type to numpy
      objects_list = objects_bboxes[0].numpy()

      scene = Scene()
      axes_obj = []

      for object_ in objects_list:

         obj = GameObject(id=int(object_[-1]), position_xywh=object_[:-1])

         # mundo id = 0, axe id = 1
         if obj.id == 0: 
            scene.mundos.append(obj)
         else:
            scene.axes.append(obj)
            axes_obj.append(obj)
   
      # if axes are present in the scene, store the scene
      if axes_obj:
         self.prev_dat.append(axes_obj)

      return scene


   def get_current_frame(self):
      return self.capture_data.frame


   def get_axe_prediction(self):
      """
      This function will return the prediction made for the axe projectile in 2D space
      The amount of previous data it will hold to account when predicting will depend
      on the frame_history integer variable
      """
      # if frame_history amount of axes hasn't been seen yet, don't predict
      if len(self.prev_dat) < self.frame_history:
         return

      # this list will store all scene data as a 2 list where first index is
      # the scene and the second index will be the centre co-ord of the axe in
      # the scene
      full_data = []
   
      # loop through scenes with axes present
      for scene in self.prev_dat.frames:

         centre_pos = []

         # loop through axes in scene and calculate their centre co-ords
         for axe_data in scene:
            centre_pos.append(axe_data.centre_cords)

         full_data.append(centre_pos)

      self.prev_dat.frames.clear()

      full_dat_len = len(full_data)

      # get frames
      current_frame = full_data[full_dat_len-1]
      after_frame = full_data[full_dat_len-2]
      last_frame = full_data[full_dat_len-3]

      # create matrix with respect to screen resolution
      grid = Grid(x=self.x, y=self.y)

      for pos1 in last_frame:
         for pos2 in after_frame:
            # get gradient of 2 points as a line
            grad = self.gradient((pos1, pos2))

            # if gradient is not vertical
            if grad:
               # add line to grid
               grid.add_line(start_pos=pos1, gradient=grad)

               # increase index
               grid.increment_projectile_index()

      pred = []

      # Todo: sort out prediction format and remember the start_pos
      for pos3 in current_frame:
         if grid.is_valid(pos3):
            prediction = (pos3)
            pred.append(prediction)

      print(grid,"\n")

      print("PREDICTION:", pred)
            
      return pred

   def parallel(self, line1, line2):
      if near(gradient:=self.gradient(line1), self.gradient(line:=line2)):
         return gradient, line

   def gradient(self, line):
      (x1, y1), (x2, y2) = line

      # Ensure that the line is not vertical
      if x1 != x2:
         gradient = (1. / (x1 - x2)) * (y1 - y2)
         return gradient   

def near(val1, val2, rtol=1e-2, atol=1e-3):
      return np.isclose(val1, val2, rtol=rtol, atol=atol)

game = GameAnalysis(weights="/home/alpharaoh/Documents/coding/Machine-Learning/Projects/MDAI/best_2.pt")