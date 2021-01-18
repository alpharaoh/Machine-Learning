"""
This program will create a visual representation of the data given by
YOLO and data_analysis
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw

AXE_COLOR = (0, 255, 0)
MUNDO_COLOR = (0, 0, 255)

class Visualisation():
   def __init__(self, x, y):
      super().__init__()
      self.x = x
      self.y = y

   def show_visuals(self, objects_in_scene, image, axe_pred):
      """
      This is the main method call to open the live capture with
      added visuals to represent the data
      """
      image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

      # draw grid (slow)
      #image = self.draw_grid(image)

      # add axe bounding box
      #image = self.return_bbox_image(image, objects_in_scene.axes, "Axe", AXE_COLOR)

      # add mundo bounding box
      #image = self.return_bbox_image(image, objects_in_scene.mundos, "Mundo", MUNDO_COLOR)

      # add a circle/dot at the centre of the axe bbox
      image = self.show_centre_of_bbox(image, objects_in_scene.axes)

      # if there is a prediction made in the current frame, draw an arrow graphic to highlight
      # where the program predicts the axe will go
      if axe_pred:
         image = self.draw_pred_arrows(image, axe_pred, 1)




      # open live capture window with new shapes
      try:
         image = cv2.resize(image, (960, 540))  
         cv2.imshow("visualisation", image)

         if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

      except:
         pass

   def show_centre_of_bbox(self, image, objects):
      """
      This function will return an image with a small circle at the centre of the object
      bounding box. This is helpful since the predictions are being based off the centre
      of the axe bounding box and it helps to visualise this
      """
      for obj in objects:
         image = cv2.circle(image, 
                           (int(obj.centre_cords[0] * self.x), int(obj.centre_cords[1] * self.y)), 
                           radius=5, 
                           color=AXE_COLOR, 
                           thickness=-1)
      
      return image


   def return_bbox_image(self, image, bboxes, label, color):
      """
      This function loops through the bounding boxes detected in the current frame
      and returns an image of drawn bounding boxes
      """
      if bboxes:
         for obj in bboxes:
            image = self.draw_single_bbox(image, obj.position_xywh, label=label, color=color)

         return image  


   def draw_single_bbox(self, image, x, color=None, label=None, line_thickness=2):
      """
      This function will return an image after it has plotted a bounding box to it. This
      function is very similiar to the one found in YOLOv5's utilities
      """

      # get pixel locations of top left and bottom right corners
      corner1 = (int(x[0] * self.x), int(x[1] * self.y))
      corner2 = (int(x[2] * self.x), int(x[3] * self.y))

      cv2.rectangle(image, corner1, corner2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

      if label:
         font_thickness = max(line_thickness - 1, 1)
         t_size = cv2.getTextSize(label, 0, fontScale= line_thickness / 3, thickness=font_thickness)[0]
         corner2 = corner1[0] + t_size[0], corner1[1] - t_size[1] - 3

         cv2.rectangle(image, corner1, corner2, color, -1, cv2.LINE_AA)  # filled

         cv2.putText(image, 
                     label, 
                     (corner1[0], corner1[1] - 2), 0, 
                     line_thickness / 3, 
                     [225, 255, 255], 
                     thickness=font_thickness, 
                     lineType=cv2.LINE_AA)

      return image

   def draw_pred_arrows(self, image, predictions, dist_mult):
      """
      TODO
      """
      print("LEN",len(predictions))

      for pred in predictions:

         start_pos, end_pos, grad = pred

         start_norm_x, start_norm_y = start_pos
         end_norm_x, end_norm_y = end_pos

         new_start_pos = (int(start_norm_x * self.x), int(start_norm_y * self.y))
         new_end_pos = (int(end_norm_x * self.x), int(end_norm_y * self.y))
         
         image = cv2.arrowedLine(image, 
                                 new_start_pos, 
                                 new_end_pos,
                                 AXE_COLOR, 
                                 2)

      return image
   
   def draw_grid(self, image, grid_size = (32, 18)):
      grid_x, grid_y = grid_size
      height, width, _ = image.shape

      step_size = int(width/grid_x)

      image = Image.fromarray(image)
      draw = ImageDraw.Draw(image)
      

      for x in range(0, width, step_size):
         line = ((x, 0), (x, height))
         draw.line(line, fill="black")

      for y in range(0, height, step_size):
         line = ((0, y), (width, y))
         draw.line(line, fill="black")

      del draw

      return np.asarray(image)
