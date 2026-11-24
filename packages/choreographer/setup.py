from setuptools import find_packages, setup

package_name = "choreographer"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO_YOUR_FULL_NAME",
    maintainer_email="TODO_YOUR_EMAIL_ADDRESS",
    description="features.json -> schedule.json choreography generator.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "generate = choreographer.generate:main",
        ],
    },
)
