from setuptools import setup

setup(
    name='LungMapUtilsExtra',
    version='1.2',
    packages=['lung_map_utils_extra'],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Duke Lungmap Team',
    description='Extra functions for image processing tasks for LungMap analysis',
    install_requires=[
        'numpy',
        'opencv-python',
        'scipy'
    ]
)
