from setuptools import setup

setup(
    name='cv2_extras',
    version='0.4',
    packages=['cv2_extras'],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Scott White',
    description='A Python library for higher level OpenCV functions used in image analysis and computer vision',
    install_requires=[
        'numpy (>=1.16)',
        'opencv-python (>=4.1)',
        'scipy (>=1.2)',
        'scikit-image (>=0.15)',
        'matplotlib (>=3.0)'
    ]
)
