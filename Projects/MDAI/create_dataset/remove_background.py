import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# alpha values included
white = [255, 255, 255, 255]
green = [64, 253, 47, 255]
green_example = [62, 255, 8, 255]

lower_green = [70, 255, 70, 255]
upper_green = [0, 255, 0, 255]

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
   x0, y0, a0 = co_ords.min(axis=0)
   x1, y1, a1 = co_ords.max(axis=0) + 1   # slices are exclusive at the top

   # crop image based on our bounding box
   cropped = image[x0:x1, y0:y1]

   return cropped

def remove_background(image: complex, background_RGBA: tuple, change_to_RGBA: tuple, crop=False):
   # convert image from BGR to RGBA
   image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).copy()

   # print(image_rgba)

   # image_rgba[green11] = [0, 0, 0, 0]

   # plt.imshow(image_rgba)
   # plt.show()
   
   # exit()

   # convert all background_RGBA coloured pixels to change_to_RGBA colour
   image_rgba[np.all(image_rgba == background_RGBA, axis=-1)] = change_to_RGBA


   #image_rgba[np.all(np.logical_and(image_rgba > lower_green, image_rgba < upper_green), axis=-1)] = change_to_RGBA


   if crop:
      return crop_image(image_rgba)

   return image_rgba

if __name__ == '__main__':
   path = "../dataset/output/frames/frame_0.png"

   img = cv2.imread(path) #returns BGR NumPy array

   final_image = remove_background(img, green_example, transparant, crop=True)
   final_image = remove_edge_noise(final_image, tol=2)
   save_image(final_image, path="../dataset/output/output_parsed_frames/", name="test2")


# if __name__ == '__main__':
#    path = "../dataset/input/example.png"
#    img = cv2.imread(path) #returns BGR NumPy array

#    final_image = remove_background(img, green, transparant, crop=True)
#    final_image = remove_edge_noise(final_image, tol=2)
#    save_image(final_image, path="../dataset/output/", name="test2")