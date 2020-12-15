import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle
#import torch


def get_normalized_bbox(x_position, y_position, width, height, image_size):
   """
   This function will locate the bounding box of foreground (e.g. Mundo) and then
   normalise the values from 0-1. This is important since YOLO uses a specific .txt format
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


def merge_background_foreground(background, foreground: list, resize_mult=0, move_position=(0, 0)):
   # get position of where the foreground should move to
   x_position, y_position = move_position

   # get x, y of foreground and background
   foreground_size = foreground.size
   background_size = background.size

   # get new x, y of foreground after resizing 
   new_size_x, new_size_y = int(foreground_size[0] * resize_mult), int(foreground_size[1] * resize_mult)
   
   # apply resize onto foreground image
   foreground_resized = foreground.resize((new_size_x, new_size_y))

   # paste resized foreground onto background at poisiton
   background.paste(
      foreground_resized, 
      (x_position, y_position), 
      foreground_resized)
   
   # get width and height of foreground image
   _, _, width, height = foreground_resized.getbbox()
   
   bbox = get_normalized_bbox(
      x_position, 
      y_position, 
      width, 
      height, 
      background_size)

   draw_bounding_box_for_testing(background, bbox)


def draw_bounding_box_for_testing(image: Image, bbox: tuple):
   plt.gca().add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor="r", facecolor="none"))

   plt.imshow(image)
   plt.show()


def stretch_for_yolo():
   pass

def save_image_with_YOLO_bb_txt():
   pass

if __name__ == '__main__':
   # example
   background_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/baron_pit_frames/frame_0.png"
   mundo_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/output_parsed_frames/mundo_500.png"

   #background_image = cv2.imread(background_path)
   #mundo_image = cv2.imread(mundo_path)
   
   background_image = Image.open(background_path)
   mundo_image = Image.open(mundo_path)

   merge_background_foreground(background_image, mundo_image, resize_mult=0.34, move_position=(600, 100))

   # plt.imshow(mundo_image)
   # plt.show()


