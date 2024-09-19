from setuptools import find_packages, setup

package_name = 'inventory_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/multilaunch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rose',
    maintainer_email='rosesareroro@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mp4_publisher = inventory_system.mp4_publisher:main',
            'object_detection = inventory_system.object_detection_node:main',
        ],
    },
)
