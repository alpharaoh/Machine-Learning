"""
This program will take a background image, and some foreground object images 
and merge these images together. It will also calculate the bounding boxes for 
each foreground object and save the output image and respective boundbox text file
in a target folder
"""

import cv2
import os
import numpy as np
import random
#import matplotlib.pyplot as plt
from PIL import Image
#from matplotlib.patches import Rectangle
#import tensorflow as tf
#import torch

print("Loaded...")

def get_normalized_bbox(x_position: int, y_position: int, width: int, height: int, image_size: tuple):
   """
   This function will locate the bounding box of foreground (e.g. Mundo) and then
   normalise the values from 0-1. This is important step to get YOLO format correct.

   YOLO doesn't use bounding boxes where xy is the top-left corner, but xy is the 
   center of the rectangle
   """
   # get background screen resolution (e.g. 1920, 1080)
   x_image_size = image_size[0]  
   y_image_size = image_size[1]

   # create min/max bounding box
   x_min = x_position
   x_max = x_position + width
   y_min = y_position
   y_max = y_position + height

   # convert minx/max to center co-ordinate bounding box
   x = int(x_min + width / 2)
   y = int(y_min + height / 2)

   # normalise 
   x /= x_image_size
   y /= y_image_size
   width /= x_image_size
   height /= y_image_size

   normalized_bbox = (x, y, width, height)

   return normalized_bbox#torch.tensor(normalized_bbox)

def get_random_position(background_size: tuple):
   """
   WIP
   """
   x_min_bound, x_max_bound = 200, background_size[0]-400
   y_min_bound, y_max_bound = 170, background_size[1]-250
   
   random_x = random.randint(x_min_bound, x_max_bound)
   random_y = random.randint(y_min_bound, y_max_bound)

   return random_x, random_y

def merge_background_foreground(background: Image, foreground: list, resize_mult=0, occurances=1):
   """
   This function will use a move_position tuple to move and paste a foreground object 
   to the background and can resize the foreground object using a resize multiplier.
   It returns the image and the foreground's bounding box.
   """
   
   # get x, y of foreground and background
   foreground_size = foreground.size
   background_size = background.size

   # get position of where the foreground should move to
   x_position, y_position = get_random_position(background_size)

   # get new x, y of foreground after resizing 
   new_size_x, new_size_y = int(foreground_size[0] * resize_mult), int(foreground_size[1] * resize_mult)
   
   # apply resize onto foreground image
   foreground_resized = foreground.resize((new_size_x, new_size_y))

   # paste resized foreground onto background at poisiton
   background.paste(
      foreground_resized, 
      (x_position, y_position), 
      foreground_resized)
   
   # get width and height of foreground image. 
   _, _, width, height = foreground_resized.getbbox()
   
   # create bounding box
   bbox = get_normalized_bbox(
      x_position, 
      y_position, 
      width, 
      height, 
      background_size)

   return background, bbox


def draw_bounding_box_for_testing(image: Image, bbox: tuple):
   """
   WIP
   """
   #plt.gca().add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor="r", facecolor="none"))
   
   image_rgba = np.array(image)
   image_rgba = cv2.cvtColor(image_rgba, cv2.COLOR_BGR2RGBA).copy()

   bbox_tensor = tf.convert_to_tensor(np.asarray(bbox + (0, )), dtype=tf.float32)
   colour = tf.convert_to_tensor(np.asarray((255, 0, 0, 255)), dtype=tf.float32)

   print(tf.rank(image_rgba), "\n\n")
   image_with_boxes = tf.image.draw_bounding_boxes(image_rgba, bbox_tensor, colour)
   print(image_with_boxes)
   plt.imshow(image_with_boxes)
   plt.show()


def stretch_for_YOLO(image: Image, size=(320,320)):
   """
   YOLO trains on square images that are multiples of 32. e.g. 320x320, 352x352, 416x416, 608x608, etc.
   """
   return image.resize(size)

def save_image_with_YOLO_bb_txt(image: Image, bbox: tuple, output_path: str, object_id: int, file_name="test"):
   """
   WIP
   """
   # get file name
   # mundo_name = mundo_path.split("/")[-1:][0].replace(".png", "")
   content = f"{object_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"

   txt_file = open(f"{output_path}/{file_name}.txt", "w")
   txt_file.write(content)
   txt_file.close()

   image.save(f"{output_path}/{file_name}.jpg")

def load_images_and_run_all(background_path_folder: str, mundo_folder_path: str, output_path: str):
   """
   WIP
   """
   for i, file_name in enumerate(os.listdir(background_path_folder)):
      for j, mundo_file_name in enumerate(os.listdir(mundo_folder_path)):

         try: #DS.Store files may be captured 
            background_image = Image.open(f"{background_path_folder}/{file_name}")
            mundo_image = Image.open(f"{mundo_folder_path}/{mundo_file_name}")
         except Exception as e:
            print(e)
         else:

            images_not_null = background_image is not None and mundo_image is not None

            if images_not_null:
               # create image
               image, bbox = merge_background_foreground(
                  background_image,
                  mundo_image, 
                  resize_mult=0.30,
                  occurances=2)
               
               save_image_with_YOLO_bb_txt(image, bbox, output_path, 1, file_name=f"final_{i}_{j}")

if __name__ == '__main__':
   # example
   #background_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/baron_pit_frames/frame_0.png"
   background_path_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/baron_pit_frames/"
   #mundo_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/output_parsed_frames/mundo_500.png"
   mundo_path_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/output_parsed_frames/mundo/"
   # output_path_annotations = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/merged_images/annotations/"
   # output_path_images = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/merged_images/images/"
   output_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/merged_images/"

   # background_image = Image.open(background_path)
   # mundo_image = Image.open(mundo_path)

   # image, bbox = merge_background_foreground(
   #    background_image,
   #    mundo_image, 
   #    resize_mult=0.34, 
   #    move_position=(600, 100))

   #save_image_with_YOLO_bb_txt(stretch_for_YOLO(image), bbox, output_path, 1)

   load_images_and_run_all(background_path_folder, mundo_path_folder, output_path)