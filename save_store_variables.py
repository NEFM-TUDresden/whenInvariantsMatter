'''
This is a supplementary material to the paper

"The role of the invariants in neural network-based modelling of incompressible hyperelasticity"
by Franz Dammass, Karl A. Kalina and Markus Kästner.

The code is provided under the CC BY-SA 4.0 license, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
When you find this code useful, please cite the corresponding paper.
'''


import pickle
import os

def save_to_file(path: str, filename: str, variable: list) -> None:
    os.makedirs(path, exist_ok=True) # generate path
    with open(os.sep.join([f"{path}",f"{filename}.pkl"]), 'wb+') as file: # store list variable in file
        pickle.dump(variable, file) 
               
def load_from_file(path: str, filename: str) -> list:
    with open(os.sep.join([f"{path}",f"{filename}.pkl"]), 'rb') as file: # get list variable from file
        variable = pickle.load(file)
    return variable
    