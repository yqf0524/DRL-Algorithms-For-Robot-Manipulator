import numpy as np


def test(a=None):
    b = a if a else 3
    print(b)


if __name__ == '__main__':
    a = np.ones((2,2))
    b = np.ones((2,2))*2
    print(a,b)
    print(np.matmul(a, b))