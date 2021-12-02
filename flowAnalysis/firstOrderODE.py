
from numericalMethods import RungeKutta
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import odeint

solver = RungeKutta()
val = solver.iterFourthOrder(0, 0, 3, 0.001)
print(val)

def dydx(y, x):
    return 2*x

def odeSolver(y0, xs):
    ys = odeint(func=dydx, y0=y0, t=xs)
    ys = np.array(ys).flatten()
    return ys

xs = np.linspace(0, 1, 10)
y0 = 0 
ys = odeSolver(y0, xs)
print(ys)

plt.plot(xs, ys)
plt.show()
