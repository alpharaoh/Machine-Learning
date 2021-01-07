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

from yolo_inference import YoloInference

os.system("clear")
print("Loaded modules.\n\nStarting...")


NAMES = ["Mundo", "Axe"]
AXE_COLOR = [0, 255, 0]
MUNDO_COLOR = [0, 0, 255]

class GameObject():
   """
   This class 
   """
   def __init__(self, id=None, position_xywh=None):
      self.id = id
      self.position_xywh = position_xywh

   # def __repr__(self):
   #    super().__repr__()
   #    return f"GameObject: {NAMES[self.id]} found at {self.position_xywh}"


class Scene():

   def __init__(self, mundos=[], axes=[]):
      self.mundos = mundos
      self.axes = axes

   def __len__(self):
      return len(self.mundos + self.axes)

   def clear(self):
      self.mundos.clear()
      self.axes.clear()


class GameAnalysis():
   
   def __init__(self, weights="."):
      assert type(weights) == str, "Weights are path of .pt weights file as Type String"

      self.prev_axe_data = None
      
      self.capture_data = YoloInference(weights)
      self.x = self.capture_data.screen_capture.x_res
      self.y = self.capture_data.screen_capture.y_res
      self.gen = self.capture_data.infer_real_time_frames()


      while True:
         current_frame_data = next(self.gen)
         objects_in_scene = self.get_state(current_frame_data)

         print(f"Total objects detected: {len(objects_in_scene)}")

         self.show_bboxes(objects_in_scene)
         

         objects_in_scene.clear()

   def show_bboxes(self, objects_in_scene):
      image = self.return_bbox_image(objects_in_scene.axes, "Axe", AXE_COLOR)
      image = self.return_bbox_image(objects_in_scene.mundos, "Mundo", MUNDO_COLOR)

      try:
         cv2.imshow("visualisation", image)

         if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

      except:
         pass


   def get_state(self, scene_data):      
      objects_bboxes = scene_data.xyxyn
      predictions = scene_data.pred
      
      objects_list = objects_bboxes[0]

      scene = Scene()

      for object_ in objects_list:
         obj = GameObject(id=int(object_[-1]), position_xywh=object_[:-1])

         if obj.id == 0: # mundo
            scene.mundos.append(obj)
         else:
            scene.axes.append(obj)
   
      return scene


   def return_bbox_image(self, bboxes, label, color):
      if bboxes:
         for obj in bboxes:
            image = self.draw_single_bbox(self.get_current_frame(), obj.position_xywh, label=label, color=color)

         return image      
   
   
   def draw_single_bbox(self, image, x, color=None, label=None, line_thickness=2):
      """Plots one bounding box on image"""

      # get pixel locations of top left and bottom right corners
      corner1, corner2 = (int(x[0] * self.x), int(x[1] * self.y)), (int(x[2] * self.x), int(x[3] * self.y))

      cv2.rectangle(image, corner1, corner2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

      if label:
         font_thickness = max(line_thickness - 1, 1)
         t_size = cv2.getTextSize(label, 0, fontScale= line_thickness / 3, thickness=font_thickness)[0]
         corner2 = corner1[0] + t_size[0], corner1[1] - t_size[1] - 3
         cv2.rectangle(image, corner1, corner2, color, -1, cv2.LINE_AA)  # filled
         cv2.putText(image, label, (corner1[0], corner1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

      image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

      return image


   def get_centre_of_bbox(self, bbox):
      pass


   def get_current_frame(self):
      return self.capture_data.frame
      


game = GameAnalysis(weights="/Users/alpharaoh/Downloads/best (1).pt")