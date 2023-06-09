from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function returns the list of requirements. 
    '''
    requirements = []
    with open(file_path) as f:
        requirements= [r.strip() for r in f.readlines()]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='recommendation-system',
    version='0.0.1',
    author='ani',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    #['pandas','numpy','seaborn'] etc
)