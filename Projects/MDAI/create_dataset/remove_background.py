import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
This program will handle parsing an image data set by removing the background and cropping 
the images of some target folder and outputting the new images in another folder
"""

#RED, BLUE, GREEN, ALPHA

#Suitable green colour bound
LOWERGREEN = [0, 240, 0, 255]
UPPERGREEN = [150, 255, 150, 255]

TRANSPARANT = [0, 0, 0, 0]

def remove_edge_noise(image, tol=2):
   """
   After removing background, there are still some (green) noise around the edges. Using the
   cv2 erode method will erode the edges to try and remove these redundant pixels
   """
   # kernal decides nature of opertaion e.i. tolerence
   kernel = np.ones((tol, tol), np.uint8)

   # erode boundary pixels
   return cv2.erode(image, kernel, iterations=1)

def crop_image(image: complex):
   """
   This function will calculate a bounding box in respect to where the edges of the image is
   and crop accordingly. 
   """
   # get value of True or False depending on if pixel should be removed or not
   mask = image > TRANSPARANT

   # find co-ordinates of non-transparant pixels 
   co_ords = np.argwhere(mask)

   # bounding box of non-transparant pixels.
   x0, y0, _ = co_ords.min(axis=0)
   x1, y1, _ = co_ords.max(axis=0) + 1   # slices are exclusive at the top

   # crop image based on our bounding box
   cropped = image[x0:x1, y0:y1]

   return cropped

def crop_outer_green_box(image, crop_amount=50): 
   """
   This function returns an image which is slightly cropped inwards from all sides. This is
   necassery since video compression in this case created dark (green) lined artifacts on 
   the edge of the image frames
   """
   # get image resolution
   y, x, _ = image.shape

   # get cropped resolution
   crop_x, crop_y = x - crop_amount, y - crop_amount

   # get start point to crop at
   start_x = x // 2 - (crop_y // 2)
   start_y = y // 2 - (crop_y // 2)

   return image[start_y:start_y + crop_y, start_x:start_x + crop_x]

def remove_background(image: complex, background_RGBA_lower: tuple, background_RGBA_upper: tuple, change_to_RGBA: tuple, crop=False):
   """
   This function will return a perfectly cropped RGBA image after changing pixels with a bounded 
   colour [background_RGBA_lower -> background_RGBA_upper] to transparant
   """
   # convert image from BGR to RGBA
   image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).copy()

   # if green in bounds
   arg = np.logical_and(image_rgba <= background_RGBA_upper, image_rgba >= background_RGBA_lower)

   # convert all background_RGBA coloured pixels to change_to_RGBA colour
   image_rgba[np.all(arg, axis=-1)] = change_to_RGBA

   if crop:
      return crop_image(image_rgba)

   return image_rgba

def load_images_and_run_all():
   """
   This method will create and save cropped and chroma keyed png images from an input dataset of 
   images with a plain coloured background 
   """
   count = 0

   for file_name in os.listdir(folder_path):

      # read all images in folder
      image = cv2.imread(os.path.join(folder_path, file_name))
      
      if image is not None:
         # create image
         final_image = crop_outer_green_box(image, crop_amount=100)
         final_image = remove_background(final_image, LOWERGREEN, UPPERGREEN, TRANSPARANT, crop=True)
         final_image = remove_edge_noise(final_image, tol=2)

         save_image(final_image, path="/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/output_parsed_frames/", name=f"mundo_{count}")
         count += 1

def save_image(image: complex, path="./", name="remove_background_image"):
   # convert NumPy array to Pillow Image
   new_im = Image.fromarray(image)
   new_im.save(f"{path}/{name}.png")

if __name__ == '__main__':
   folder_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/frames/"
   load_images_and_run_all()