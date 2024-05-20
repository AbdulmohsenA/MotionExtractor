import cv2
import torchvision
from tqdm import tqdm

class MotionExtractor:
    
    def __init__(self, verbose=False):
        self.verbose = verbose
 
    def process(self, filename, buffer_size=20, ratio=0.5, output_file='output.mp4'):
        
        reader = torchvision.io.VideoReader(filename, "video")
        c, height, width = next(reader)['data'].shape
        fps = round(reader.get_metadata()["video"]["fps"][0])
        duration = reader.get_metadata()["video"]["duration"][0]
        n_frames = int(duration * fps)
        buffer = []

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        if self.verbose:
            print(f"Height x Width: {height}x{width}")
            print(f"FPS: {fps}, Duration: {duration:.2f} seconds.")
            print(f"Total Frames: {n_frames} Frames")
        
        # Fill the buffer window with frames
        # TODO: Big time consumption to convert torch to numpy, optimize.
        buffer = [next(reader)['data'].permute(1, 2, 0).numpy() for _ in range(buffer_size)]

        for frame in tqdm(reader, total=n_frames - buffer_size - 1, desc="Processing Frames", ncols=100):
            
            new_frame = frame['data'].permute(1, 2, 0).numpy()
            old_inverse_frame = 255 - buffer[0]
            motion_frame = cv2.addWeighted(old_inverse_frame, ratio, new_frame, 1 - ratio, 1)
            
            writer.write(motion_frame)
            
            # shift the buffer window 1 frame to the right
            buffer.pop(0)
            buffer.append(new_frame)
        
        writer.release()


