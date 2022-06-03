import numpy as np


def standard_normal(x, n=1):
    return x + np.random.normal(size=(n,))


def get_normal(sigma):
    def normal(x, n=1):
        return x + np.random.normal(0, sigma, size=(n,))

    return normal


def linear_binary(x, n=1):
    """Returns 1 or -1 with probability p=(x+1)/2 and 1-p respectively."""
    assert -1 <= x <= 1, f"error with x: {x}"
    x = (x + 1) / 2
    ret = np.array([-1, 1])[np.random.binomial(1, x, size=(n,))]
    return ret
