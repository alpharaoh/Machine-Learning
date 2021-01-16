import cv2
import numpy as np

"""
This class holds data about an object in the scene such as:
- the position
- the identifier of the object
- the centre co-ordinates
"""
class GameObject():
  
    def __init__(self, id=None, position_xywh=None):
        self.id = id
        self.position_xywh = position_xywh
        self.centre_cords = self.get_centre_of_bbox(position_xywh)

    def get_centre_of_bbox(self, bbox):
        x1, y1, x2, y2, _ = bbox

        centre_x = (x2-x1)/2 + x1
        centre_y = (y2-y1)/2 + y1

        return (centre_x, centre_y)


"""
The grid class is how the program will represent the game data and use it for 
predictions etc. It works by splitting the screen into many quadrants and 
each quadrant will store information (an integer).

Take for example the following grid. Where the 0 represents no data and the 1 represent the gradient
between 2 axes. This can be useful since we are able to easily work out if this is a valid projectile
as we can compare a third axe to see if its position is at 0 or 1. If it is a 1, it is valid and the bot
must avoid the 1's

[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0]]
"""
class Grid():

    def __init__(self, x=1920, y=1080, ratio=(16, 9), ratio_mult=2, line_tol=1):
        self.x, self.y = x, y

        # line tolerance represents how thick the line (of 1's for example) will be
        # where a higher tolerance means a thicker line
        self.line_tol = line_tol

        self.ratio_x, self.ratio_y = ratio
        assert type(ratio_mult) == int, "ratio multiplier must be of type int"

        # calculate the max squares in each dimension
        self.squares_x = self.ratio_x * ratio_mult
        self.squares_y = self.ratio_y * ratio_mult

        # projectile index is the number which will be changed in the grid
        # there can be different representations in the grid. e.g. 1 is the
        # the axes, 2 is the mundos, 3 is the safe areas etc.
        self.projectile_index = 1

        self.grid = None

        self.create_grid()

    def __repr__(self):
        return str(self.grid)

    def increment_projectile_index(self):
        self.projectile_index += 1

    def clear_grid(self):
        self.create_grid()

    def create_grid(self):

        assert (self.ratio_x / self.x)  == (self.ratio_y / self.y), f"ratio must be respective of resolution; \
            {self.ratio_x}:{self.ratio_y} does not corrispond to the resolution {self.x}x{self.y}"

        self.grid = np.zeros((self.squares_y, self.squares_x), dtype=int)

    def change_value(self, pos, value):
        x, y = pos
        self.grid[y, x] = value

    def add_xleftright(self, y, x):
        """
        This adds extra quadrants surrounding the gradient line depending on the tolerance
        """
        self.change_value((x, y), self.projectile_index)

        # make sure that there is no out of bound error
        for i in range(self.line_tol):

            addition_max = x + i + 1
            subtraction_min = x - i - 1

            x_bounds = addition_max >= 0 and addition_max < self.squares_x
            y_bounds = subtraction_min >= 0 and subtraction_min < self.squares_y 

            if x_bounds and y_bounds:

                # add quadrant(s) to right of point
                self.grid[y, addition_max] = self.projectile_index
                # add quadrant(s) to left of point
                self.grid[y, subtraction_min] = self.projectile_index


    def add_line(self, start_pos, gradient):
        """
        This will use the equation of a line;
        y = mx + c
        where y is y index, m is the gradient, x is the x index and c is the y-intercept
        to fill out a gradient on the grid using a start position
        """
        x, y = start_pos
        gradient = np.float64(gradient)

       # print(f"Gradient: {gradient}\nx: {x}\ny: {y}")


        # get y-intercept (c) using a rearranged version of y = mx + c
        c = y - (gradient * x)

        for y_new in range(int(y+1), len(self.grid)):
            # get x using a rearranged version of y = mx + c
            x = (y_new - c) / gradient

            # handles case when x is outside of the number of squares on the grid we have
            if abs(x) >= self.squares_x:
                break

            self.add_xleftright(y_new, int(x))

    def value_in_projectile(self, x_grid, y_grid):
        """
        compares the position of an axe and returns True if it is located on a 1.
        """
        return 1 == self.grid[y_grid, x_grid]

    def change_square(self, pos=(400, 300)):
        pos_x, pos_y = pos

        grid_y, grid_x = self.res_to_grid_squares(pos_x, pos_y)

        self.grid[grid_y, grid_x] = self.projectile_index

    def res_to_grid_squares(self, x, y): 
        """
        turns position to a position on the grid
        e.g. if max resolution is 1920x1080 with 32x18 grid and the position is 400x400
        the grid position of 400x400 respective of 1920x1080 is (5, 7)
        """
        # get ratio of pos to max res
        rat_x, rat_y = x/self.x, y/self.y 

        # times ratio with max grid resolution to get square index
        # of position x, y
        grid_x = round(rat_x * self.squares_x) - 1
        grid_y = round(rat_y * self.squares_y) - 1

        return grid_x, grid_y

    def O1_2_xy(self, x, y):
        """
        This takes a normalised x, y that ranges from 0-1
        to 2 new values that range from the resolution of the
        image, e.g. from 0-1920 and 0-1080
        """

        new_x = round(x*self.x)
        new_y = round(y*self.y)

        return new_x, new_y

"""
This class stores the previous data such as previous scenes
"""
class previousFrameData():
    def __init__(self):
        self.frames = []

    def append(self, scene):
        self.frames.append(scene)

    def show_all(self):
            for j, axe in enumerate(self.frames[0].axes):
                print(f"Axe {j+1} - {axe.centre_cords}")
    
    def __len__(self):
        return len(self.frames)

"""
This class holds any instances of type Object and splits them
up occordingly depending on if the object is a mundo or axe
"""
class Scene():
    
    def __init__(self, mundos=[], axes=[]):
        self.mundos = mundos
        self.axes = axes

    def __len__(self):
        return len(self.mundos) + len(self.axes)

    def clear(self):
        self.mundos.clear()
        self.axes.clear()
