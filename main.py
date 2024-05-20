import sys
from motionextractor import motion_extractor

def main():

    filename = 'downsampled.mp4'
    processor = motion_extractor.MotionExtractor(verbose=True)
    processor.process(filename)

if __name__ == "__main__":
    main()