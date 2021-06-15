import pathlib

from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='torch-simplify',
    version='0.0.9',
    description='Simplification of pruned models for accelerated inference',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/EIDOSlab/simplify',
    author='EIDOSlab',
    author_email='eidoslab@di.unito.it',
    license='BSD 3',
    packages=find_packages(exclude=('profile', '.github', '.idea')),
    zip_safe=False,
    install_requires=[
        'numpy==1.20.3',
        'Pillow==8.2.0',
        'tabulate==0.8.9',
        'torch==1.8.1',
        'torchvision==0.9.1',
        'typing-extensions==3.10.0.0',
    ],
    python_requires='>=3.6'
)
