from setuptools import setup, find_packages

setup(
    name="motion_extractor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torchvision',
        'tqdm',
        'opencv-python'
        ],
    url="https://github.com/AbdulmohsenA/MotionExtractor",
)
