"""
This is the main controller for the Mundo character. These functions allow the AI 
to perform actions in game.
"""

import pynput.mouse
from pynput.mouse import Button
import pynput.keyboard
import time

class MundoController():
   def __init__(self, screen_dimensions=(1920, 1080), ):
      self.screen_dimensions = screen_dimensions
      self.mouse = pynput.mouse.Controller()
      self.keyboard = pynput.keyboard.Controller()

   def mouse_click(self, primary: bool, amount=1):
      if primary: 
         pynput.mouse.Controller().click(pynput.mouse.Button.right, 1)
      else: 
         pynput.mouse.Controller().click(pynput.mouse.Button.left, 1)

   def point_click_move(self, des_position: tuple):
      self.mouse.position = des_position
      self.mouse_click(False)

   def point_use_ability(self, des_position_on_screen: tuple, ability_key="q"):
      self.mouse.position = des_position_on_screen
      self.keyboard.press(ability_key)

if __name__ == '__main__':
   mundo = MundoController()
   time.sleep(2)

   #mundo.point_click((600,600))
   mundo.point_use_ability((600,600))
   #mundo.buy_control_ward()
