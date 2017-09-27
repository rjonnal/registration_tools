import time
import numpy as np
from numba import jit


@jit
def do(a):
    for k in range(10):
        a = np.fft.fft(a)
        amax = a.max()
        astd = a.std()

if __name__=='__main__':
    a = np.random.randn(1024,1024)
    do(a)
