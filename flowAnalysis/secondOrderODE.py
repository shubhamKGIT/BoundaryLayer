from scipy.integrate import solve_ivp, odeint
import numpy as np
import matplotlib.pyplot as plt 

def dU_dx(U, x):
    "dy_dx for second order ODE"
    # here solving for y'' + 2y' + y = 0
    # z = y' 
    # z' + 2*z + y = 0
    # y = U[0], z = U[1]
    # returns [z, -y + 2*z]
    return [U[1], -U[0] - 2*U[1] + 1]

def solver(U0, xs):
    "takes the xs and values for U0 and return ys"
    Us = odeint(dU_dx, U0, xs)
    # Us = np.array(Us).flatten()
    return Us

U0 = [0, -1]   #[y(0), y'(0)]
xs = np.linspace(0, 10, 100)
print(f"xs \n {xs}")
Us = solver(U0, xs)
print(f"Us \n {Us}")
ys = Us[:, 0]   # first column is y, second is y' or z
plt.plot(xs, Us)
plt.show()