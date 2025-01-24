from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(filepath:str)->List[str]:
    '''
    This function will return requirements
    '''
    requirements = []
    with open(filepath) as file_object:
        requirements=file_object.readlines()
        requirements = [line.replace("\n", "") for line in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='ICP',
    version='0.0.1',
    author='heinhtutzaw',
    author_email="hein.zaw@stonybrook.edu",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)