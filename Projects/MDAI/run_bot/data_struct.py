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
        y, x = pos

        x_bounds = x < self.squares_x or x >= 0
        y_bounds = y < self.squares_y or y >= 0

        assert x_bounds and y_bounds, f"GridBoundsError: Your values Y:{y} or X:{x} not in the bounds Y:{self.squares_y} and X:{self.squares_x}"

        self.grid[y, x] = value

    def add_xleftright(self, y, x):
        """
        This adds extra quadrants surrounding the gradient line depending on the tolerance
        """
        self.change_value((y, x), self.projectile_index)
        
        for i in range(self.line_tol):

            addition_max = x + i + 1
            subtraction_min = x - i - 1

            in_x_bounds = addition_max >= 0 and addition_max < self.squares_x
            in_y_bounds = subtraction_min >= 0 and subtraction_min < self.squares_y


            # make sure that there is no out of bound error
            if in_x_bounds:
                pos = (y, addition_max)

                # add quadrant(s) to right of point
                self.change_value(pos, self.projectile_index)

            if in_y_bounds:
                pos = (y, subtraction_min)

                # add quadrant(s) to left of point
                self.change_value(pos, self.projectile_index)

    def add_xleftright_mundo(self, y, x, second_grid):
        safe_grid_spaces = []
        
        self.change_value((y, x), self.direction_index)

        max_range = self.line_tol

        for i in range(max_range):

            addition_max = x + i + 1
            subtraction_min = x - i - 1

            in_x_bounds = addition_max >= 0 and addition_max < self.squares_x
            in_y_bounds = subtraction_min >= 0 and subtraction_min < self.squares_y

            # make sure that there is no out of bound error
            if in_x_bounds:
                pos = (y, addition_max)

                bounds_x = addition_max + 1 < self.squares_x and addition_max + 1 >= 0

                if bounds_x:
                    found_in_second_grid = second_grid[y, addition_max] == 1

                    at_max_pos_x_range = addition_max == x + max_range

                    space_available = second_grid[y, addition_max + 1] == 0

                    if found_in_second_grid and at_max_pos_x_range and space_available:
                        safe_grid_spaces.append(y, addition_max + 1)

                # add quadrant(s) to right of point
                self.change_value(pos, self.direction_index)

            if in_y_bounds:
                pos = (y, subtraction_min)

                bounds_x = addition_max + 1 < self.squares_x and addition_max + 1 >= 0

                if bounds_x:
                    found_in_second_grid = second_grid[y, subtraction_min] == 1

                    at_max_neg_x_range = subtraction_min == x - max_range - 2

                    space_available = second_grid[y, subtraction_min - 1] == 0

                    if found_in_second_grid and at_max_neg_x_range and space_available:
                        safe_grid_spaces.append(y, subtraction_min - 1)

                # add quadrant(s) to left of point
                self.change_value(pos, self.direction_index)
            
        return safe_grid_spaces


    def add_line(self, start_pos=(0, 0), gradient=None, direction=False):
        """
        This will use the equation of a line;
        y = mx + c
        where y is y index, m is the gradient, x is the x index and c is the y-intercept
        to fill out a gradient on the grid using a start position
        """
        x, y = start_pos

        # get y-intercept (c) using a rearranged version of y = mx + c
        c = y - (gradient * x)

        safe_zones = []

        for y_new in range(int(y+1), len(self.grid)):
            # get x using a rearranged version of y = mx + c
            x = (y_new - c) / gradient

            # handles case when x is outside of the number of squares on the grid we have
            if abs(x) >= self.squares_x:
                break
            
            if direction:
                self.direction_index = 1

                safe_zone = self.add_xleftright_mundo(y_new, int(x))
                safe_zones.append(safe_zone)

                self.direction_index += 1
            else:
                self.add_xleftright(y_new, int(x))

        if direction:
            return safe_zones

    def value_in_projectile(self, x_grid, y_grid):
        """
        compares the position of an axe and returns True if it is located on a 1.
        """
        return self.grid[y_grid, x_grid] != 0

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
        grid_x = round(rat_x * self.squares_x)
        grid_y = round(rat_y * self.squares_y)

        return grid_x, grid_y

    def norm_2_xy(self, x, y):
        """
        This takes a normalised x, y that ranges from 0-1
        to 2 new values that range from the resolution of the
        image, e.g. from 0-1920 and 0-1080
        """
        new_x = round(x*self.x)
        new_y = round(y*self.y)

        return new_x, new_y

    def is_valid(self, position):
        x, y = position

        new_x, new_y = self.norm_2_xy(x, y)

        grid_x, grid_y = self.res_to_grid_squares(new_x, new_y)

        is_valid = self.value_in_projectile(grid_x, grid_y)

        self.change_value((grid_x, grid_y), 9)

        return is_valid

    def add_mundo(self, bbox):
        x, y, w, h, _ = np.array(bbox)

        x = int(x*self.squares_x)
        y = int(y*self.squares_y)
        x2 = int(w*self.squares_x)
        y2 = int(h*self.squares_y)
        
        for grid_x in range(x, x2):
            for grid_y in range(y-2, y2):

                x_bounds = grid_x < self.squares_x and grid_x >= 0
                y_bounds = grid_y < self.squares_y and grid_y >= 0

                if x_bounds and y_bounds:
                    self.grid[grid_y, grid_x] = 1
    
    def obj_in_mundo_grid(self, positions):
        obj_in_mundo_grid = False
        
        for position in positions:
            x, y = position
            new_x, new_y = self.norm_2_xy(x, y)
            grid_x, grid_y = self.res_to_grid_squares(new_x, new_y)

            if self.grid[grid_y, grid_x] != 0:
                obj_in_mundo_grid = True

        return obj_in_mundo_grid

"""
This class stores the previous data such as previous scenes
"""
class previousFrameData():
    def __init__(self):
        self.frames = []

    def append(self, axes):
        self.frames.append(axes)

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
