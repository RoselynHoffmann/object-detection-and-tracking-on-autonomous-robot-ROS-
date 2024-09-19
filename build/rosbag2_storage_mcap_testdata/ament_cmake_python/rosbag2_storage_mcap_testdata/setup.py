from setuptools import find_packages
from setuptools import setup

setup(
    name='rosbag2_storage_mcap_testdata',
    version='0.15.12',
    packages=find_packages(
        include=('rosbag2_storage_mcap_testdata', 'rosbag2_storage_mcap_testdata.*')),
)
