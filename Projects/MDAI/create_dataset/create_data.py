"""
This program will take a background image, and some foreground object images 
and merge these images together. It will also calculate the bounding boxes for 
each foreground object and save the output image and respective boundbox text file
in a target folder
"""

#10-13

import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
#import tensorflow as tf
#import torch

print("Loaded...")

class CreateDataset():
   def __init__(self, background_path_folder, mundo_path_folder, axe_path_folder, output_path):
      self.x_mins = []
      self.y_mins = []
      self.x_maxes = []
      self.y_maxes = []
      self.widths = []
      self.heights = []
      self.background_path_folder = background_path_folder
      self.mundo_path_folder = mundo_path_folder
      self.axe_path_folder = axe_path_folder
      self.output_path = output_path

   def get_normalized_bbox(self, x_position: int, y_position: int, width: int, height: int, image_size: tuple):
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

      self.x_mins.append(x_min) 
      self.y_mins.append(y_min)
      self.x_maxes.append(x_max)
      self.y_maxes.append(y_max)

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

   def get_random_position(self, background_size: tuple):
      """
      WIP
      """
      # x_min_bound, x_max_bound = 200, background_size[0]-400
      # y_min_bound, y_max_bound = 170, background_size[1]-250

      x_min_bound, x_max_bound = 500, 1300
      y_min_bound, y_max_bound = 280, 930
      
      random_x = random.randint(x_min_bound, x_max_bound)
      random_y = random.randint(y_min_bound, y_max_bound)

      return random_x, random_y

   def merge_background_foreground(self, background: Image, foreground: list, resize_mult=0, count=0, bbox_list=[]):
      """
      This function will use a move_position tuple to move and paste a foreground object 
      to the background and can resize the foreground object using a resize multiplier.
      It returns the image and the foreground's bounding box.

      The function is recursive depending on the size of foreground
      """

      # get image we are going to add to background image
      foreground_object = foreground[count]

      # get x, y of foreground and background
      foreground_size = foreground_object.size
      background_size = background.size

      # get position of where the foreground should move to
      x_position, y_position = self.get_random_position(background_size)

      # get new x, y of foreground after resizing 
      new_size_x, new_size_y = int(foreground_size[0] * resize_mult), int(foreground_size[1] * resize_mult)
      
      # apply resize onto foreground image
      foreground_resized = foreground_object.resize((new_size_x, new_size_y))

      # paste resized foreground onto background at poisiton
      background.paste(
         foreground_resized, 
         (x_position, y_position), 
         foreground_resized)
      
      # get width and height of foreground image. 
      _, _, width, height = foreground_resized.getbbox()
      
      self.widths.append(width)
      self.heights.append(height)

      # create bounding box
      bbox = self.get_normalized_bbox(
         x_position, 
         y_position, 
         width, 
         height, 
         background_size)

      # save our bounding box
      bbox_list.append(bbox)

      # run function again if we havn't looped through all images in foreground
      if count < len(foreground)-1:
         # incrememnt count
         self.merge_background_foreground(background, foreground, resize_mult=resize_mult, count=count+1, bbox_list=bbox_list)

      return background, bbox_list


   def draw_bounding_box_for_testing(self, image: Image):
      """
      WIP
      """

      plt.imshow(image)

      for i in range(len(self.widths)):
         plt.gca().add_patch(Rectangle((self.x_mins[i], self.y_mins[i], self.x_maxes[i], self.y_maxes[i]), self.widths[i], self.heights[i], linewidth=1, edgecolor="r", facecolor="none"))

      plt.show()


   def stretch_for_YOLO(self, image: Image, size=(320,320)):
      """
      YOLO trains on square images that are multiples of 32. e.g. 320x320, 352x352, 416x416, 608x608, etc.
      """
      return image.resize(size)

   def save_image_with_YOLO_bb_txt(self, image: Image, bbox: tuple, output_path: str, object_id: int, file_name="test"):
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

   def load_images_and_run_all(self, occurances=1):
      """
      WIP
      """
      # loop through background path folder to get file names
      for i, file_name in enumerate(os.listdir(self.background_path_folder)):

         # loop through mundo images path folder to get file names
         for j, mundo_file_name in enumerate(os.listdir(self.mundo_path_folder)):

            # get size of axe images
            axe_sizes = len(os.listdir(axe_path_folder))-1

            # since there are less images of axe we use mod to keep looping through the axe images
            axe_file_name = f"axe_{j % axe_sizes}.png"

            try: #DS.Store files may be captured 
               background_image = Image.open(f"{self.background_path_folder}/{file_name}")
               mundo_image = Image.open(f"{self.mundo_path_folder}/{mundo_file_name}")
               axe_image = Image.open(f"{self.axe_path_folder}/{axe_file_name}")

            except Exception as e:
               print(e)

            else:
               images_not_null = background_image is not None and mundo_image is not None

               # since we are using .paste() method, the background image will be the final image
               # image is a clearer variable here
               image = background_image

               if images_not_null:
                  # e.g. if there is a mundo and axe images we want to paste, 
                  # if occurances is 2, there will be 2 mundos and 2 axes 
                  for _ in range(occurances):
                     # create image
                     image, bboxes = self.merge_background_foreground(
                        image,
                        [mundo_image, axe_image], 
                        resize_mult=0.30)

                  print(bboxes, "\n", len(bboxes))
                     
                  self.draw_bounding_box_for_testing(image)

                  #save_image_with_YOLO_bb_txt(image, bbox, output_path, 1, file_name=f"final_{i}_{j}")

if __name__ == '__main__':
   background_path_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/baron_pit_frames/"
   mundo_path_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/output_parsed_frames/mundo/"
   axe_path_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/output_parsed_frames/axe/"
   output_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/merged_images/"

   #save_image_with_YOLO_bb_txt(stretch_for_YOLO(image), bbox, output_path, 1)
   create = CreateDataset(background_path_folder, mundo_path_folder, axe_path_folder, output_path)

   create.load_images_and_run_all(occurances=2)