from setuptools import find_packages, setup

package_name = 'VoiceClassification'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lars',
    maintainer_email='zhcv321741185@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    
    entry_points={
        'console_scripts': [
            'voice_buffer_node = VoiceClassification.voice_buffer_node:main',
            'inference_node    = VoiceClassification.inference_node:main',
            'mic_node          = VoiceClassification.mic_node:main',
            'odas_bridge_node  = VoiceClassification.odas_bridge_node:main',
            'odas_audio_bridge_node = VoiceClassification.odas_audio_bridge_node:main',
            'window_recorder_node = VoiceClassification.window_recorder_node:main',
            'respeaker_doa_node = VoiceClassification.respeaker_doa_node:main',
        ],
    },

)
