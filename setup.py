import pathlib

from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

__version__ = "1.1.3"

setup(
    name='torch-simplify',
    version=__version__,
    description='Simplification of pruned models for accelerated inference',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/EIDOSlab/simplify',
    author='EIDOSlab',
    author_email='eidoslab@di.unito.it',
    license='BSD 3-Clause',
    packages=find_packages(exclude=('profile', '.github', '.idea', 'tests', "debug_env")),
    zip_safe=False,
    install_requires=[
        'torch',
        'torchvision',
    ],
    python_requires='>=3.6'
)
