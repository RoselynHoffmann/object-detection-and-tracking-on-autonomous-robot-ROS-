## Introduction
This repository takes the previous code for the object detection node on the unitree go 2 and modifies it to use an appearance based object tracker for inventory counting. The appearance based object tracker uses cosine similarity to update object tracks. To run the example with the ros_bag.

## Prerequisites
Operating System: Ubuntu 18.04 or 20.04 (compatible with ROS Melodic or Noetic)
ROS (Robot Operating System): Melodic or Noetic
Python: 3.6 or higher
CUDA-compatible GPU: Optional but recommended for performance
ROS Bag File: Contains video frames published on the /video_frames topic

## Installation
### 1. Install Ros Noetic
ROS Noetic (Ubuntu 20.04): http://wiki.ros.org/noetic/Installation/Ubuntu

### 2. Set up ROS work_space

# Install miniconda to create workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh
source ~/.bashrc

# verify installation
conda --version

# setup environment
conda install mamba -c conda-forge
mamba create -n ros_env python=3.11
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults

# Create the catkin workspace directory
mkdir -p ~/catkin_ws/src

# Navigate to the workspace directory
cd ~/catkin_ws/

# Initialize the workspace
catkin_make

# Navigate to the src directory
cd ~/catkin_ws/src

# Clone your repository 
git clone https://github.com/RoselynHoffmann/object-detection-and-tracking-on-autonomous-robot-ROS- inventory_system

# run dependency commands
sudo apt-get update
sudo apt-get install -y ros-noetic-cv-bridge ros-noetic-sensor-msgs
pip install opencv-python numpy torch torchvision ultralytics scipy

# Source the setup file
source devel/setup.bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

# launch program
catkin_make
roslaunch inventory_system inventory_system.launch

