import os
from sklearn.model_selection import train_test_split
from shutil import copy

print("Loaded.")

class HandleDataset():
   def __init__(self, dataset_folder, target_folder):
      self.dataset_folder = dataset_folder
      self.target_folder = target_folder
      self.labels = os.listdir(f"{dataset_folder}/labels")
      self.images = os.listdir(f"{dataset_folder}/images")
      self.train_clothing = None
      self.val_clothing = None

   def split_train_test(self, test_size):
      self.train_clothing, self.val_clothing = train_test_split(self.images, test_size=test_size)

   def create_yolo_dataset(self):
      for filename in self.train_clothing:
         copy(f"{self.dataset_folder}/images/{filename}", f"{self.target_folder}/images/train/")
         copy(f"{self.dataset_folder}/labels/{self.get_filename(filename)}.txt", f"{self.target_folder}/labels/train/")
      
      for filename in self.val_clothing:
         copy(f"{self.dataset_folder}/images/{filename}", f"{self.target_folder}/images/val/")
         copy(f"{self.dataset_folder}/labels/{self.get_filename(filename)}.txt", f"{self.target_folder}/labels/val/")

   def get_filename(self, filename):
      return filename.split(".")[0]

if __name__ == '__main__':
   dataset_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/mundododgeball/full_dataset_v2/"
   target_folder = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/mundododgeball/yolo_dataset_v2/"

   ds = HandleDataset(dataset_folder, target_folder)

   ds.split_train_test(0.1)

   ds.create_yolo_dataset()


