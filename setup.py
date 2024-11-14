from setuptools import find_packages,setup
from typing import List
HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[i.replace("\n","") for i in requirements]
        
    return requirements







setup(
    name="mlproject",
    version='0.0.1',
    author='Kaustubh Kaushik',
    author_email='kaustubhkaushik26@gmail.com',
    packages=find_packages(),
    install_packages=get_requirements('requirements.txt')
)