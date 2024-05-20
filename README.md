# Motion Extractor

A Python module to extract motion from videos.

## Why

- To collect pure motion out of videos for ML models.
- To empower the effect of motion in videos.

# Usage
```python
from motion_extractor import MotionExtractor

extractor = MotionExtractor(verbose=True)
extractor.process('input_video.mp4', buffer_size=20, ratio=0.5, output_file='output.mp4')
```

# Notes

- `buffer_size`: The amount of timesteps to calculate the difference of motion.
- 'ratio': The weight out output between the first and last frame. 0.5 means all have same weight. i.e. only pure motion will be extracted.

---

## TODO
- Optimize the process
- Make the process use torchvision instead of cv2