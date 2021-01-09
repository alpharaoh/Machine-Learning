import numpy as np

squares_x = 18
squares_y = 32

x, y = 1920, 1080

pos = (400, 300)

pos_x, pos_y = pos

grid_x, grid_y = pos_x/x, pos_y/y 

grid_x = int(grid_x * squares_x) - 1
grid_y = int(grid_y * squares_y) - 1

print(grid_x, grid_y)

grid = np.zeros((squares_x, squares_y), dtype=int)

grid[grid_y, grid_x] = 1


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