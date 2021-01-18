import numpy as np
import cv2
from PIL import Image, ImageDraw


def draw_grid(image, grid_size = (64, 36)):
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

      return image 

image = cv2.imread("./screenshot.png")

image = draw_grid(image)

image = np.array(image)
cv2.imshow("", image)

cv2.waitKey(0)


exit()
squares_x = 32
squares_y = 18

x, y = 1920, 1080

# pos = (400, 300)

# pos_x, pos_y = pos

# grid_x, grid_y = pos_x/x, pos_y/y 

# grid_x = int(grid_x * squares_x) - 1
# grid_y = int(grid_y * squares_y) - 1


# grid = np.zeros((squares_x, squares_y), dtype=int)

# grid[grid_y, grid_x] = 1


# print(grid)

value = 1

grid = np.zeros((squares_y, squares_x), dtype=int)


######
def cover_grid(y, x, tol):
   grid[y, x] = value
   print(x, y)

   for i in range(tol):
      addition_max = x + i + 1
      subtraction_min = x - i - 1

      x_bounds = addition_max >= 0 and addition_max < squares_x
      y_bounds = subtraction_min >= 0 and subtraction_min < squares_x 

      if x_bounds and y_bounds:
         grid[y, addition_max] = value
         grid[y, subtraction_min] = value

pos = (17, 1)
grad = 1

# y = mx + c

x, y = pos

#grid[y, x] = 1

c = y - (grad * x)
print("\n")

# x = (y - c) / m

tol = 1

for y_new in range(y+1, len(grid)):
   x = (y_new - c) / grad

   if abs(x) >= 32:
      break

   cover_grid(y_new, int(x), tol)

print(grid)







# current_frame = [(5, 10), (2, 3)]
# after_curr_frame = [(20, 2), (3,5), (4,10)]
# last_frame = [(-1, -5)]

# lines

# for i in after_curr_frame:
#    for j in last_frame:
      


# l1 = [(-1,-5), (3,5)]
# l2 = [(3,5), (5, 10)]
# l3 = [(-0.5, -4.5), (3.5, 5.5)]

# def gradient(l):
#     """Returns gradient 'm' of a line"""
#     m = None
#     # Ensure that the line is not vertical
#     if l[0][0] != l[1][0]:
#         m = (1./(l[0][0]-l[1][0]))*(l[0][1] - l[1][1])
#         return m

# def parallel(l1,l2):
#    if gradient(l1) != gradient(l2):
#       return

#    print(gradient(l1), l1)

# a = parallel(l1, l3)
# print(a)





# # vectors1 = []
# # vectors2 = []

# # for after_axes in after_curr_frame:

# #    for last_axes in last_frame:
# #       v1 = last_axes[0] - after_axes[0]
# #       v2 = last_axes[1] - after_axes[1]
# #       try:
# #          vectors1.append([last_axes, round(v1/v2, 2)])
# #       except:
# #          pass

      
# # for current_axes in current_frame:
   
# #    for after_axes in after_curr_frame:
# #       v1 = after_axes[0] - current_axes[0]
# #       v2 = after_axes[1] - current_axes[1]
# #       try:
# #          vectors2.append(round(v1/v2, 2))
# #       except:
# #          pass

# # print("1",vectors1)
# # print("2",vectors2)

# # found_vectors = set(vectors1[1]).intersection(vectors2)

# # vectors_and_start_pos = []

# # for vector in vectors1:
# #    for j in found_vectors:
# #       if j == vector[1]:
# #          vectors_and_start_pos.append([i[0], j])