"""
This program will take a background image, and some foreground object images 
and merge these images together. It will also calculate the bounding boxes for 
each foreground object and save the output image and respective boundbox text 
file in a target folder
"""
import cv2
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

from add_filters import ImageFilters

print("Loaded.")

MUNDO_ID = 0 
AXE_ID = 1

class CreateDataset():
   def __init__(self, background_path_folder, mundo_path_folder, axe_path_folder, output_path_images, output_path_labels, baronpit_bbox_path, cursor_path_folder, start_pos=0, image_size=None):
      self.ids = []
      self.x_mins = []
      self.y_mins = []
      self.widths = []
      self.bboxes = []
      self.x_maxes = []
      self.y_maxes = []
      self.heights = []

      self.flip = False
      self.is_cursor = False

      self.start_pos = start_pos
      self.image_size = image_size
      self.axe_path_folder = axe_path_folder
      self.mundo_path_folder = mundo_path_folder
      self.output_path_images = output_path_images
      self.cursor_path_folder = cursor_path_folder
      self.output_path_labels = output_path_labels
      self.background_path_folder = background_path_folder

      self.filters = ImageFilters(width=image_size[0])

      self.current_background_image_index = 0
      
      self.baron_bboxes = self.get_baron_bbox(baronpit_bbox_path)
      

   def get_baron_bbox(self, baronpit_bbox_path):
      """
      This function will take in a path to text file that has bounding boxes
      of each background image and return a list of it's contents
      """
      # open text file at path location
      baronpit_bbox_file = open(baronpit_bbox_path, "r")
      
      # get contents
      baronpit_content = baronpit_bbox_file.read().strip().split("\n")
      
      # split data into each image
      list_of_bbox_for_baronpit = [bbox.split(", ") for bbox in baronpit_content]

      baronpit_bbox_file.close()
     
      return list_of_bbox_for_baronpit

   def paste_cursor(self, image):
      """
      This function returns an image that has been altered to add 1-3 cursors. The 
      cursors have a chance to be a normal default cursor, or a frame from the cursor
      clicking animation
      """
      # chance to have more than 1 cursors
      cursor_amount = random.randint(1, 3)

      cursors = []

      for _ in range(cursor_amount):
         chance = random.randint(0, 2)

         # 66.6% chance to get a default cursor 
         if chance <= 1:
            cursor_image = Image.open(f"{self.cursor_path_folder}/default.png")
            cursors.append(cursor_image)

         # otherwise get a random cursor animation out of 6 images
         else:
            random_cursor_index = random.randint(1, 6)
            cursor_anim_image = Image.open(f"{self.cursor_path_folder}/{random_cursor_index}.png")
            cursors.append(cursor_anim_image)

      for cursor in cursors:
         x_position, y_position = self.get_random_position()
         
         image.paste(
            cursor, 
            (x_position, y_position), 
            cursor)

      return image

   def get_normalized_bbox(self, x_position: int, y_position: int, width: int, height: int, image_size: tuple):
      """
      This function will locate the bounding box of foreground (e.g. Mundo) and then
      normalise the values from 0-1. This is important step to get YOLO format correct.

      YOLO doesn't use bounding boxes where xy is the top-left corner, but xy is the 
      center of the rectangle
      """
      # get screen resolution
      size = self.image_size[0]

      # create min/max bounding box
      x_min = x_position
      x_max = x_position + width
      y_min = y_position
      y_max = y_position + height

      # if resizing, stretch bbox
      if not self.image_size == None:
         x_min, x_max, y_min, y_max = self.filters.bbox_stretch_for_YOLO(
            width, height, 
            (x_min, x_max, y_min, y_max), 
            size=self.image_size)

            
         # stretch width height to allign with new bbox
         width *= self.filters.stretch_factor_x
         height *= self.filters.stretch_factor_y

      if self.flip:
         x_min, x_max, y_min, y_max = self.filters.flip_bbox(x_min, x_max, y_min, y_max, width)

      self.x_mins.append(x_min) 
      self.y_mins.append(y_min)
      self.x_maxes.append(x_max)
      self.y_maxes.append(y_max)

      # convert minx/max to center co-ordinate bounding box
      x = int(x_min + width / 2)
      y = int(y_min + height / 2)

      # normalise 
      x /= size
      y /= size
      width_norm = width / size
      height_norm = height / size

      normalized_bbox = (x, y, width_norm, height_norm)

      return width, height, normalized_bbox


   def get_random_position(self):
      """
      This function will get the bounding box for the current backround image and return 
      a random position inside the bounding box
      """

      # get bounding box of current current background image
      x_min_bound, y_min_bound, x_max_bound, y_max_bound  = [int(value) for value in self.baron_bboxes[self.current_background_image_index]]
      
      # get a random xy value between it's max and min values
      random_x = random.randint(x_min_bound, x_max_bound)
      random_y = random.randint(y_min_bound, y_max_bound)

      return random_x, random_y


   def merge_background_foreground(self, background: Image, foreground: list, resize_mult=0, count=0):
      """
      This function will use a move_position tuple to move and paste a foreground 
      object to the background and can resize the foreground object using a resize 
      multiplier. It returns the image and the foreground's bounding box.

      The function is recursive depending on the size of foreground
      """

      # get image we are going to add to background image
      foreground_object = foreground[count][0]

      # get the object id - needed for identifying the object in bounding box
      object_id = foreground[count][1]

      if object_id == None:
         self.is_cursor = True

      # get x, y of foreground and background
      foreground_size = foreground_object.size
      background_size = background.size

      # get position of where the foreground should move to
      x_position, y_position = self.get_random_position()

      # get new x, y of foreground after resizing 
      new_size_x, new_size_y = int(foreground_size[0] * resize_mult), int(foreground_size[1] * resize_mult)
      
      # apply resize onto foreground image
      foreground_resized = foreground_object.resize((new_size_x, new_size_y))

      # paste resized foreground onto background at poisiton
      background.paste(
         foreground_resized, 
         (x_position, y_position), 
         foreground_resized)
      
      if not self.is_cursor:
         # get width and height of foreground image. 
         _, _, width, height = foreground_resized.getbbox()
         
         # create bounding box
         width, height, bbox = self.get_normalized_bbox(
            x_position, 
            y_position, 
            width, 
            height, 
            background_size)

         # save our new values
         self.ids.append(object_id)
         self.bboxes.append(bbox)
         self.widths.append(width)
         self.heights.append(height)

      # run function again if we havn't looped through all images in foreground
      if count < len(foreground)-1:
         # incrememnt count
         self.merge_background_foreground(background, foreground, resize_mult=resize_mult, count=count+1)

      return background

   def prob_flip(self):
      """
      This method chooses whether the current image will be flipped or not. This is based
      on a 10% chance 
      """
      random_num = random.randint(1, 10)

      if random_num == 1:
         self.flip = True

   def draw_bounding_box_for_testing(self, image: Image):
      """
      This method will show the bounding boxes around each object image in the background 
      image. This is used for testing purposes.
      """
      plt.imshow(image)

      # loop through amount of images in foreground and add a rectangle to it
      for i in range(len(self.widths)):
         plt.gca().add_patch(Rectangle((self.x_mins[i], self.y_mins[i], self.x_maxes[i], self.y_maxes[i]), 
                                        self.widths[i], self.heights[i], 
                                        linewidth=1, edgecolor="r", facecolor="none"))
      plt.show()


   def cleanup(self):
      """
      This method will clear all values from the last image. This is needed for when
      we are displaying the bounding boxes so to make sure there are no redundant
      bounding boxes from the last image on the new image  
      """
      self.ids, self.x_mins, self.y_mins, self.x_maxes, self.y_maxes, self.widths, self.heights, self.bboxes = [[] for i in range(8)]


   def save_image_with_YOLO_bb_txt(self, image: Image, file_name="test"):
      """
      This method creates a YOLO formatted text file for the bounding boxes and 
      respective id's and also saves the final images and the text file in the 
      target folder
      """
      txt_file = open(f"{self.output_path_labels}/{file_name}.txt", "w")

      # loop through amount of images in foreground
      for i, bbox in enumerate(self.bboxes):
         content = f"{self.ids[i]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
         txt_file.write(content)

      txt_file.close()

      image.save(f"{self.output_path_images}/{file_name}.jpg")


   def get_random_images(self, mundo_images_amount: int, axe_images_amount: int):
      """
      This function returns a random mundo image and a random axe image
      """
      random_mundo_index = random.randint(0, mundo_images_amount)
      random_axe_index = random.randint(0, axe_images_amount)

      random_mundo_image = Image.open(f"{self.mundo_path_folder}/mundo_{random_mundo_index}.png")
      random_axe_image = Image.open(f"{self.axe_path_folder}/axe_{random_axe_index}.png")

      return random_mundo_image, random_axe_image


   def images_to_use_with_probability(self, full_images: list):
      """
      40% chance to get all 4 images
      30% chance to get 3 images
      20% chance to get 2 images
      10% chance to get 1 image
      """

      # get a random number from 1 to 100
      random_num = random.randint(1, 100)

      if random_num <= 10:
         return [full_images[0]]
      
      elif random_num > 10 and random_num <= 30:
         return [full_images[0], full_images[2]]
      
      elif random_num > 30 and random_num <= 60:
         return [full_images[0], full_images[1], full_images[2]]

      else:
         return full_images


   def load_images_and_run_all(self, occurances=1):
      """
      This method will loop through all background and foreground images and run
      merge the images.

      You can also specify the occurances of each merge i.e. you can multiply how many
      foreground objects will appear on the image
      """
      # loop through background path folder to get file names
      for i, file_name in enumerate(os.listdir(self.background_path_folder)):

         self.current_background_image_index = i

         # loop through mundo images path folder to get file names
         for j, axe_file_name in enumerate(os.listdir(self.axe_path_folder)):

            # get size of axe and mundo images
            axe_images_amount = len(os.listdir(self.axe_path_folder))-1
            mundo_images_amount = len(os.listdir(self.mundo_path_folder))-2 #-2 here since there is a DSStore file

            # since there are less images of mundos we use mod to keep looping through the axe images
            mundo_file_name = f"mundo_{j % mundo_images_amount}.png"

            try: #DS.Store files may be captured 
               background_image = Image.open(f"{self.background_path_folder}/{file_name}")
               mundo_image = Image.open(f"{self.mundo_path_folder}/{mundo_file_name}")
               axe_image = Image.open(f"{self.axe_path_folder}/{axe_file_name}")

               random_mundo_image, random_axe_image = self.get_random_images(mundo_images_amount, axe_images_amount)

            except Exception as e:
               continue

            else:
               # since we are using paste method, the background image will be the final image
               # image is a clearer variable here
               image = background_image
               
               self.prob_flip()

               # e.g. if there is a mundo and axe images we want to paste, 
               # if occurances is 2, there will be 2 mundos and 2 axes 
               for _ in range(occurances):
                  # create image
                  image = self.merge_background_foreground(
                     image,
                     self.images_to_use_with_probability([[mundo_image, MUNDO_ID], [random_mundo_image, MUNDO_ID], [axe_image, AXE_ID], [random_axe_image, AXE_ID]]),
                     resize_mult=0.28)

               image = self.paste_cursor(image)
               image = self.filters.image_stretch_for_YOLO(image, size=self.image_size)
               image = self.filters.prob_filtered_image(image)

               if self.flip:
                  image = self.filters.flip_image(image)
                  self.flip = False
               
               # self.draw_bounding_box_for_testing(image)
               
               self.save_image_with_YOLO_bb_txt(image, file_name=f"final_{i}_{j + self.start_pos}")

               self.cleanup()


if __name__ == '__main__':
   background_path_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/baron_pit_frames/"
   mundo_path_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/output_parsed_frames/mundo/"
   axe_path_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/output_parsed_frames/axe/"
   cursor_path_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/cursor/cropped_cursor_frames"
   output_path_image = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/mundododgeball/full_dataset_v2/images/"
   output_path_labels = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/mundododgeball/full_dataset_v2/labels/"
   baronpit_bbox_file_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/baron_pit_frames/baron_pit_bbox.txt"

   create = CreateDataset(background_path_folder, mundo_path_folder, axe_path_folder, output_path_image, output_path_labels, baronpit_bbox_file_path, cursor_path_folder, image_size=(1920, 1080), start_pos=452)

   create.load_images_and_run_all()
