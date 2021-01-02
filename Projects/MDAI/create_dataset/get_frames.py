"""
This program takes in a video and splits it up into frames and saves these 
images in a target folder
"""
import cv2

# before running this method, we should remove duplicate frames using FFmpeg 
# command: ffmpeg -i input.mp4 -vf mpdecimate,setpts=N/FRAME_RATE/TB out.mp4

def save_all_frames(video_path: str, video_name: str, output_path: str, file_name: str, starting_count=0):
   video_capture = cv2.VideoCapture(f"{video_path}/{video_name}")

   # read frame data
   success, image = video_capture.read()

   count = starting_count

   # if success = False, this means that the video has ended
   while success:
      # save image
      cv2.imwrite(f"{output_path}/{file_name}_{count}.png", image)
      success, image = video_capture.read()
      
      # increase count so that files don't get overwritten
      count += 1


if __name__ == '__main__':
   video_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/input/teemogg_vid_capture"
   video_name = "mundo_throw_ff.mp4"
   output_path = "/Users/alpharaoh/Documents HDD/Machine Learning/Machine-Learning/Projects/MDAI/dataset/output/new_pictures/mundo/"

   save_all_frames(video_path, video_name, output_path, "mundo", starting_count=0)