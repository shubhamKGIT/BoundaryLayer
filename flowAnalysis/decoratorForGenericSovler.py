from numpy import array
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

def numericalPython(someSolver):
    "wrapper functions for solving any ode, wraps the nuemrical method"
    def solveAndPlot(dUdx, U0, xs):
        "wrapped function here"
        ys = someSolver(dUdx, U0, xs)
        print("applied the wrapped function")
        plt.plot(xs, ys)

    return solveAndPlot

def dy_dx(y, x):
    return 2*x   # some quadratic fn for testing


@numericalPython
def odesolver(dy_dx, y0, xs):
    "takes dy_dx, y0, xs and returns ys using the method defined here"
    return solve_ivp(dy_dx, y0, xs)

odesolver(dy_dx, 0, [1, 2, 3])