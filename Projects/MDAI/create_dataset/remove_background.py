import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# alpha values included
green_example = [62, 255, 8, 255]

lower_green = [0, 240, 0, 255]
upper_green = [150, 255, 150, 255]

transparant = [0, 0, 0, 0]

def save_image(image: complex, path="./", name="remove_background_image"):
   # convert NumPy array to Pillow Image
   new_im = Image.fromarray(image)
   new_im.save(f"{path}/{name}.png")

def remove_edge_noise(image, tol=2):
   # kernal decides nature of opertaion e.i. tolerence
   kernel = np.ones((tol, tol), np.uint8)

   # erode boundary pixels
   return cv2.erode(image, kernel, iterations=1)

def crop_image(image: complex):
   # get value of True or False depending on if pixel should be removed or not
   mask = image > transparant

   # find co-ordinates of non-transparant pixels 
   co_ords = np.argwhere(mask)

   # bounding box of non-transparant pixels.
   x0, y0, _ = co_ords.min(axis=0)
   x1, y1, _ = co_ords.max(axis=0) + 1   # slices are exclusive at the top

   # crop image based on our bounding box
   cropped = image[x0:x1, y0:y1]

   return cropped

def remove_background(image: complex, background_RGBA_lower: tuple, background_RGBA_upper: tuple, change_to_RGBA: tuple, crop=False):
   # convert image from BGR to RGBA
   image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).copy()

   # if green in bounds
   arg = np.logical_and(image_rgba <= background_RGBA_upper, image_rgba >= background_RGBA_lower)

   # convert all background_RGBA coloured pixels to change_to_RGBA colour
   image_rgba[np.all(arg, axis=-1)] = change_to_RGBA

   if crop:
      return crop_image(image_rgba)

   return image_rgba

# there are dark green lines outside the frames that ruins the main crop
def crop_outer_green_box(image, crop_amount=50): 
   # get image resolution
   y, x, _ = image.shape

   # get cropped resolution
   crop_x, crop_y = x - crop_amount, y - crop_amount

   # get start point to crop at
   start_x = x // 2 - (crop_y // 2)
   start_y = y // 2 - (crop_y // 2)

   return image[start_y:start_y + crop_y, start_x:start_x + crop_x]

def load_images_and_run_all():
   count = 0

   for file_name in os.listdir(folder_path):

      # read all images in folder
      image = cv2.imread(os.path.join(folder_path, file_name))
      
      if image is not None:
         # create image
         final_image = crop_outer_green_box(image, crop_amount=100)
         final_image = remove_background(final_image, lower_green, upper_green, transparant, crop=True)
         final_image = remove_edge_noise(final_image, tol=2)

         save_image(final_image, path="/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/output_parsed_frames/", name=f"mundo_{count}")
         count += 1

if __name__ == '__main__':
   folder_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/frames/"
   load_images_and_run_all()