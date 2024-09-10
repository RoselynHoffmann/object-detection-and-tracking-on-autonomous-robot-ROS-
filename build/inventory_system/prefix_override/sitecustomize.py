import sys
if sys.prefix == '/home/rose/miniconda3/envs/ros2_ws':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rose/ros2_ws/install/inventory_system'
