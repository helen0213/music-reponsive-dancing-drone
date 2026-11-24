from glob import glob
import os

from setuptools import find_packages, setup

package_name = "show_runner"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO_YOUR_FULL_NAME",
    maintainer_email="TODO_YOUR_EMAIL_ADDRESS",
    description="Safety gate and schedule player for the dancing Duckiedrone show.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "safety_node = show_runner.safety_node:main",
            "player_node = show_runner.player_node:main",
        ],
    },
)
