import sys
import cv2
import torchvision
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm


if len(sys.argv) != 2:
    print("Usage: python script.py <video_file>")
    sys.exit(1)

FILENAME = sys.argv[1]
BUFFER_SIZE = 20 # Frames of change of motion


reader = torchvision.io.VideoReader(FILENAME, "video")

c, h, w = next(reader)['data'].shape
fps = round(reader.get_metadata()["video"]["fps"][0])
duration = reader.get_metadata()["video"]["duration"][0]
n_frames = int(duration * fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('./output.mp4', fourcc, fps * 2, (w, h))

print(f"HeightxWidth: {h}x{w}")
print(f"FPS: {fps}, Duration: {duration:.2f} seconds.")
print(f"Total Frames: {n_frames} Frames")

# Fill the buffer window with frames
buffer = [next(reader)['data'].permute(1, 2, 0).numpy() for _ in range(BUFFER_SIZE)]

for index, frame in enumerate(tqdm(reader, total=n_frames - BUFFER_SIZE - 1, desc="Processing Frames", ncols=100)):
    
    new_frame = frame['data'].permute(1, 2, 0).numpy()
    
    old_inverse_frame = 255 - buffer[0]
    
    gamma = 0 if index < 100 else 2
    
    motion_frame = cv2.addWeighted(old_inverse_frame, 0.4, new_frame, 0.6, gamma)
    
    writer.write(motion_frame)

    # shift the buffer window 1 frame to the right
    buffer.pop(0)
    buffer.append(new_frame)
    
writer.release()