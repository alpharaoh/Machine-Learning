import numpy as np
import matplotlib.pyplot as plt
import cv2
from .remove_background import crop_image

path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/frames/frame_1.png"

lower_green = [0, 255, 0, 255]
upper_green = [150, 255, 150, 255]

image = cv2.imread(path) #returns BGR NumPy array
image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).copy()

arg = np.logical_and(image_rgba <= upper_green, image_rgba >= lower_green)

image_rgba[np.all(arg, axis=-1)] = [0,0,0,0]

a = crop_image(image_rgba)

#print(image_rgba)
plt.imshow(a)
plt.show()