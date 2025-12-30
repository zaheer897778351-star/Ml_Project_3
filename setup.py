from setuptools import find_packages,setup
from typing import List

dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    req = []
    with open(file_path) as file_obj:
        req = file_obj.readline()
        req = [i.replace("\n","") for i in req]
        if dot in req:
            req.remove(dot)
        
    return req

setup(
    name= 'Ml_project_3',
    version='0.0.1',
    author='Zaheer_Ahmad',
    author_email='zaheer897778351@gmail.com',
    packages=find_packages(),
    install_requriments = get_requirements('req2.txt')
)