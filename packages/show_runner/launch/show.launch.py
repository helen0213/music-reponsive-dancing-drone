#!/usr/bin/env python3
"""Start safety_node, primitives_node and player_node.

ros2 launch show_runner show.launch.py \
    schedule_path:=/path/to/schedule.json config_path:=/path/to/assets/config

config_path is a DIRECTORY containing limits.yaml and latency.yaml.
Audio is not a parameter: the music is played externally (phone) and the
laptop sends the start signal on show/start_in.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PROJECT = os.environ.get("DT_PROJECT_PATH", ".")


def generate_launch_description():
    def as_str(name):  # keep paths (or an empty log_dir) from being parsed as YAML
        return ParameterValue(LaunchConfiguration(name), value_type=str)

    schedule_path = as_str("schedule_path")
    config_path = as_str("config_path")
    limits_file = PathJoinSubstitution([LaunchConfiguration("config_path"), "limits.yaml"])
    hover_altitude = ParameterValue(LaunchConfiguration("hover_altitude"), value_type=float)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "schedule_path",
                default_value=os.path.join(PROJECT, "assets", "schemas", "schedule.example.json"),
                description="schedule.json to play (default: the example schedule).",
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value=os.path.join(PROJECT, "assets", "config"),
                description="Directory containing limits.yaml and latency.yaml.",
            ),
            DeclareLaunchArgument("hover_altitude", default_value="1.0", description="Takeoff altitude, m."),
            DeclareLaunchArgument("log_dir", default_value="", description="Where the player writes its CSV log."),
            Node(
                package="show_runner",
                executable="safety_node",
                name="safety_node",
                output="screen",
                emulate_tty=True,
                parameters=[{"config_path": ParameterValue(limits_file, value_type=str), "hover_altitude": hover_altitude}],
            ),
            Node(
                package="drone_primitives",
                executable="primitives_node",
                name="primitives_node",
                output="screen",
                emulate_tty=True,
                parameters=[{"config_path": ParameterValue(limits_file, value_type=str), "hover_altitude": hover_altitude}],
            ),
            Node(
                package="show_runner",
                executable="player_node",
                name="player_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "schedule_path": schedule_path,
                        "config_path": config_path,
                        "log_dir": as_str("log_dir"),
                    }
                ],
            ),
        ]
    )
