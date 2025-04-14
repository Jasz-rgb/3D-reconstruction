import numpy as np

def essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    return E

