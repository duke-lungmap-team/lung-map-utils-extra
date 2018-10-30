from setuptools import setup

setup(
    name='cv2_extras',
    version='0.2',
    packages=['cv2_extras'],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Scott White',
    description='A Python library for higher level OpenCV functions used in image analysis and computer vision',
    install_requires=[
        'numpy',
        'opencv-python',
        'scipy'
    ]
)
