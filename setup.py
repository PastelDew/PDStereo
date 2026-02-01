"""
The build/compilations setup
>> pip install -r requirements.txt
>> python setup.py install
"""
import pip
import logging
import os
import platform

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


REQUIREMENTS_PATH = 'requirements.txt'

def _parse_requirements():
    global REQUIREMENTS_PATH
    with open(REQUIREMENTS_PATH) as f:
        requirements = f.read().strip().split('\n')
        return requirements


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements()
except Exception as err:
    logging.warning('Fail load requirements file, so using default ones.')
    logging.warning(err)
    install_reqs = []

setup(
    name='PDStereo',
    version='1.0',
    url='https://github.com/PastelDew/PDStereo',
    author='PastelDew',
    author_email='pasteldew@gmail.com',
    license='GPL-3.0',
    description='The program that is for detecting based on stereo images by using Mask R-CNN.',
    packages=["mrcnn", "PDStereo", "PDStereo.Camera", "PDStereo.InjeAI", "PDStereo.QtApp"],
    install_requires=install_reqs,
    include_package_data=True,
    python_requires='>=3.4',
    long_description="""This project is for Deep learning using Mask R-CNN.
    In this project, we use RGB-D images based on stereo image.""",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords="stereo callibration matching image instance segmentation object detection mask rcnn r-cnn tensorflow keras",
)