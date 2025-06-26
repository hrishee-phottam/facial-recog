from setuptools import setup, find_packages

setup(
    name="facerecog",
    version="0.1.0",
    packages=find_packages('src'),
    package_dir={'facerecog': 'src'},
    install_requires=[
        'requests>=2.31.0',
        'pymongo>=4.5.0'
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'facerecog=facerecog.main:main',
        ],
    },
    package_data={'facerecog': ['*.py']},
    include_package_data=True,
)
