from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='inventory_system',
            executable='video_publisher',
            name='video_publisher_node'
        ),
        Node(
            package='inventory_system',
            executable='object_detection',
            name='object_detection_node'
        )
    ])
