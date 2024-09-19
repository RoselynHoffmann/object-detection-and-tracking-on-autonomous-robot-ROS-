from setuptools import find_packages
from setuptools import setup

setup(
    name='rosbag2_interfaces',
    version='0.15.12',
    packages=find_packages(
        include=('rosbag2_interfaces', 'rosbag2_interfaces.*')),
)
