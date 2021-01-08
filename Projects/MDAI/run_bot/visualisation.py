import cv2
import numpy as np

AXE_COLOR = (0, 255, 0)
MUNDO_COLOR = (0, 0, 255)

class Visualisation():
   def __init__(self, x, y):
      super().__init__()
      self.x = x
      self.y = y

   def show_visuals(self, objects_in_scene, image, axe_pred):
      image = self.return_bbox_image(image, objects_in_scene.axes, "Axe", AXE_COLOR)
      image = self.return_bbox_image(image, objects_in_scene.mundos, "Mundo", MUNDO_COLOR)

      image = self.show_centre_of_bbox(image, objects_in_scene.axes)

      if axe_pred != None:
         image = self.draw_pred_arrows(image, axe_pred, 100)

      try:
         cv2.imshow("visualisation", image)

         if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

      except:
         pass

   def show_centre_of_bbox(self, image, objects):
      
      for obj in objects:
         image = cv2.circle(image, 
                           (int(obj.centre_cords[0] * self.x), int(obj.centre_cords[1] * self.y)), 
                           radius=5, 
                           color=AXE_COLOR, 
                           thickness=-1)
      
      return image


   def return_bbox_image(self, image, bboxes, label, color):
      if bboxes:
         for obj in bboxes:
            image = self.draw_single_bbox(image, obj.position_xywh, label=label, color=color)

         return image  


   def draw_single_bbox(self, image, x, color=None, label=None, line_thickness=2):
      """Plots one bounding box on image"""

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

      image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

      return image

   def draw_pred_arrows(self, image, predictions, dist_mult):
      print(predictions)

      for pred in predictions:

         start_pos, vector = pred

         end_pos = (start_pos[0] + (dist_mult *  vector), start_pos[0] + (dist_mult *  vector))

         image = cv2.arrowedLine(image, 
                                 start_pos, 
                                 end_pos,
                                 AXE_COLOR, 
                                 5)

      return image