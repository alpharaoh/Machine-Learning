"""
This program handles image filters and is able to add noise and blur to an image
It is also used to stretch images and bboxes to YOLO format
"""
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import random

class ImageFilters():
   def __init__(self, width=416):
      self.stretch_factor_x = None
      self.stretch_factor_y = None
      self.width = width
   
   def prob_filtered_image(self, image: Image):
      """
      This function returns an image that could have a chance to have filters on it

      80% chance to get no filters
      10% chance to get one filter
      10% chance to all filters (blur/noise)
      """
      # pick random number from 1 to 100
      random_num = random.randint(0, 100)

      if random_num <= 80:
         # return image with no changes
         return image

      elif random_num > 80 and random_num <= 90:
         # return image with at least one filter (50-50 chance)
         blur_or_noise = random.randint(0, 1)

         if blur_or_noise == 1:
            return self.get_blur_image(image, radius=1)
         else:
            return self.get_noisy_image(image)
         
      else:
         # return image with both filters
         image = self.get_blur_image(image, radius=1)
         image = self.get_noisy_image(image)

         return image

   def get_blur_image(self, image: Image, radius=2):
      """
      This function will return a blurred image using the GaussianBlur function
      from ImageFilter
      """
      blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
  
      return blurred_image

   def get_noisy_image(self, image: Image):
      """
      This function will return an image with a noise filter applied to it
      """
      image = np.asarray(image, dtype=np.uint8).copy()

      mean = 250
      std = 400

      # get a image of guassuan noise that has same size of image
      noise = np.random.normal(mean, std, image.shape)

      # normalise noise image so we can safely add it to the target image
      noise = cv2.normalize(src=noise, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

      # convert numpy to Pillow RGBA for overlay
      noise = Image.fromarray(np.uint8(noise)).convert("RGBA")
      image = Image.fromarray(np.uint8(image)).convert("RGBA")

      # get number from 0.1 to 0.45
      overlay_intensity = random.randint(10, 45) / 100

      # overlay noise onto image
      final_image = Image.blend(image, noise, overlay_intensity)

      # convert image back to RGB
      final_image = final_image.convert("RGB")

      return final_image
   
   def image_stretch_for_YOLO(self, image: Image, size=(320,320)):
      """
      YOLO trains on square images that are multiples of 32. e.g. 320x320, 
      352x352, 416x416, 608x608, etc.
      """
      return image.resize(size)
   
   def bbox_stretch_for_YOLO(self, width, height, bbox, size=(320, 320), image_x=1920, image_y=1080):
      """
      This function returns a min/max bounding box that has been stretched
      the same amount as the image.
      """
      pixel_size = size[0]

      self.stretch_factor_x = pixel_size / image_x
      self.stretch_factor_y = pixel_size / image_y

      x_min = round(bbox[0]*self.stretch_factor_x)
      x_max = round(bbox[1]*self.stretch_factor_x)
      y_min = round(bbox[2]*self.stretch_factor_y)
      y_max = round(bbox[3]*self.stretch_factor_y)

      return x_min, x_max, y_min, y_max

   def flip_bbox(self, x_min, x_max, y_min, y_max, width):
      """
      This function returns a bounding box after being flipped in the center
      """
      # find center of image
      middle = self.width / 2

      # calculate position of x values reflected from center 
      x_min += (middle - x_min) * 2
      x_max += (middle - x_max) * 2

      return (x_max, x_min, y_min, y_max)

   def flip_image(self, image):
         return ImageOps.mirror(image)
      