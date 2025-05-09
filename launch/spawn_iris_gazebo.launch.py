# File: /root/flymind_ws/launch/spawn_iris_gazebo.launch.py (inside Docker)

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    gazebo_ros_pkg_share = get_package_share_directory('gazebo_ros')
    gazebo_launch_file = os.path.join(
        gazebo_ros_pkg_share, 'launch', 'gazebo.launch.py'
    )

    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'iris_drone',
            # *** MODIFIED PATH FOR INSIDE DOCKER ***
            '-file', '/root/flymind_ws/simulation_models/iris/iris.sdf',
            '-x', '0', '-y', '0', '-z', '0.2',
            '-R', '0', '-P', '0', '-Y', '0'
        ],
        output='screen'
    )

    start_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={'pause': 'true'}.items()
    )

    return LaunchDescription([
        start_gazebo_cmd,
        spawn_entity_cmd,
    ])
