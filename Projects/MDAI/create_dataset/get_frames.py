import cv2

# before running this method, we should remove duplicate frames using FFmpeg 
# command: ffmpeg -i input.mp4 -vf mpdecimate,setpts=N/FRAME_RATE/TB out.mp4

def save_all_frames(video_path: str, video_name: str, output_path: str):
   video_capture = cv2.VideoCapture(f"{video_path}/{video_name}")

   # read frame data
   success, image = video_capture.read()

   count = 0

   # if success = False, this means that the video has ended
   while success:
      # save image
      cv2.imwrite(f"{output_path}/frame_{count}.png", image)
      success, image = video_capture.read()
      
      # increase count so that files don't get overwritten
      count += 1


if __name__ == '__main__':
   video_path = "../dataset/input/teemogg_vid_capture/"
   video_name = "baron_pit.mp4"
   output_path = "../dataset/output/baron_pit_frames/"

   save_all_frames(video_path, video_name, output_path)