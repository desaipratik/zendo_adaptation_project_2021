import numpy as np


def get_random_connections(dim, frac=0.5):
    """
    returns np array of binaries with p(1) = frac
    In:
        dim: int; dimension
        frac: float \in (0, 1); proportion of ones
    Out:
        dim x dim; numpy binary array
    """
    connect = np.random.binomial(1, frac, size=(dim, dim))
    """only keep upper triangle matrix (without diagonal)"""
    connect = np.triu(connect, 1)
    return connect

print(get_random_connections(4))
