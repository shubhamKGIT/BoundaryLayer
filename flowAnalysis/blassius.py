import numpy as np
from numpy.core.fromnumeric import size
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 16})
FIGSIZE = (8, 12)

"""
Following are some features of the code below: 
0. BLASSIUS METHOD IMPLEMENTATION
1. This program is name to solve the laminar boundary layer problem numerically. 
2. We use similarity solution to get a single ODE eqution form the PDE equation of BL. 
3. Instead of using the Runge-Kutta method for forward integration, we use the standard implementation from scipy module for numerical integration. 
4. The Boundary value problem is converted to initial value problem usign value for h to solver for f, g, h
5. Displacement thickness, Momentum Thickness, Shape Factor, Skin Friction Coeffs have been derived. 

source(s) for methodology: 
1. Introductory Aerodynamics and Hyderodynamics of Wing Bodies: A Software Based Approach, Frederich O. Smetana (1997), AIAA Education Series 
2. Boundary Layers, A. D. Young (1989), AIAA Education Series

"""

def dU_dx(U, x):
    """
    Define custom dy/dx to solve the ode.
    inputs: [f, f', f''] and x (x not used here by dy/dx but needed generally in ode)
    returns: [f', f'', f'''] or [f', g', h'] or [U(1), U(2), h'], where h' = func(U)
    """
    # for boundary layer, from similarity solution:
    # ff'' + 2*f''' = 0 , where f = (func(eta))
    # f' = g; g' = h; 
    # h' = -0.5* f*h  (***final ODE solved using this***)
    # U[0] = f, U[1] = g, U[2] = h 
    return [U[1], U[2], -0.5*U[0]*U[2]]

def solver(U0, x):
    " U0 contains BCs and x -> eta, which is normalised distance normal to wall"
    Us = odeint(dU_dx, U0, x)   # using standard solver from python scipy 
    return Us   # has info on f, g, h


eta = np.linspace(0, 6, 120)
# BCs ::: eta =0: f = 0, f' = 0; eta = inf: f' =1
# Need to convert BC problem to initial value problem
# with f = 0 , g = 0, value for h is used from Howarth(1931)
U0 = [0, 0, 0.33206]
Us = solver(U0, eta)   # calling the solver for f, f', f'', f''' on eta values
# print(Us)
print(f"Point where f'(eta) = 0.99: eta = {np.interp(0.99, Us[:, 1], eta)}")

def get_f_plot(Us, eta):
    "plotting BL similarity eqn numerical solution \n"
    label = ["f", "f'", "f''"]
    fig = plt.figure(figsize=FIGSIZE)
    plt.style.use('seaborn-poster')
    plt.plot(eta, Us, label= label)
    plt.title("Velocity and gradients in Boundary Layer")
    plt.xlabel("$\eta$")
    plt.ylabel("f, f', f''")
    plt.legend()
    plt.axvline(x = 4.91, color = 'black', linestyle = "--")
    plt.text(5, 1.5,"BL edge, f' = 0.99", rotation=90, fontsize=14)
    plt.show()

# get_f_plot(Us, eta)

# Getting other BL properties
mu = 1.81e-5
rho = 1.225
nu = mu/rho 
V_inf = 5
u = V_inf*Us[:, 1]
x = np.linspace(0, 0.5, 100)  # assuming 0.5 m long plate
tao = mu*V_inf*np.sqrt(V_inf/ (nu*(x)))*Us[0, 2]
Cf = 2*tao/ (rho*V_inf*V_inf)
# print(tao)
def get_cf_plot():
    fig = plt.figure(figsize=FIGSIZE)
    plt.plot(x, Cf)
    plt.title("Friction Coeff vs. x")
    plt.xlabel("x")
    plt.ylabel("Cf")
    plt.show()

get_cf_plot()

x_y_norm = np.sqrt(V_inf/ (nu*(x + 1e-9)))   # 1e-9 added to manage divide by 0
del_ = 4.91 / x_y_norm
del_star = 1.7208 / x_y_norm
theta = 0.664 / x_y_norm
shape_factor = del_star/ (theta + 1e-6)   # 1e-6 added to manage divide by 0
Fs = [del_, del_star, theta]
# print(Fs)
print(f"Shape factor (H): {np.mean(shape_factor)}")
def plot_BL_props():
    label = ["$\delta$", "$\delta$*", "$\Theta$"]
    F = np.stack(Fs, axis=1)
    fig = plt.figure(figsize=FIGSIZE)
    plt.plot(x, F, label=label)
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("$\delta$, $\delta$*, $\Theta$ (m)")
    plt.title("Laminar Boundary Layer Properties - Flat Plate")
    plt.text(0.3, 0.003, "H = $\delta$*/ $\Theta$ = 2.56")
    plt.show()

# plot_BL_props()


class Polhaussen:
    "considers uni-parametric solution, u/ue = F(eta) + lambda*G(eta)"
    def __init__(self, press_grad, V_inf, rho):
        self.mu = 1.81e-5
        self.rho = 1.225
        self.nu = self.mu/self.rho
        self.Ue, self.dUe_dx, self.d2Ue_dx2 = self.potential_flow_soln(press_grad, V_inf, rho)

    def potential_flow_soln(self, pressure_grad, V_inf, rho):
        # supposed to solve potential flow in the region outside the BL
        dp_dx = pressure_grad    # say of the order of 0.1*rho*V_inf**2 [Pa/m]
        Ue = V_inf
        dUe_dx = - dp_dx / (rho*Ue)
        d2Ue_dx2 = 0
        return Ue, dUe_dx, d2Ue_dx2

    def dD_dx(self, D, x):
        # u1/U = A*eta + B*eta^2 + C*eta^3 + D*eta^4
        # F = 2*eta - 2*eta^3 + eta^4 
        # G = (1/6)*eta*((1-eta)^3)
        # dZ/dx = h(lambda)* d2Ue/dx2 * Z^2 + g(lambda)/ Ue
        # Z = del_^2 / nu 
        # SOLVING FIRST ORDER ODE for boundary layer thickness, del_
        eps = 1e-9
        lambda_ = (D**2 / self.nu)*(self.dUe_dx)
        term1 = (self.nu/(self.Ue*(D + eps)))*(2 + (1/6)*lambda_)
        term2 = (D/ self.Ue)*((3/10) - (1/120)*lambda_)*self.dUe_dx
        term3 = (2*D/ self.Ue)*((37/315) - (1/945)*lambda_ - (1/9072)*(lambda_**2))*self.dUe_dx
        term4 = (1/945)*(D**3 /self.nu)*(self.d2Ue_dx2)
        term5 = (2/9072)*(D**5 / self.nu**2)*self.dUe_dx*self.d2Ue_dx2
        term6 = ((37/ 315) - (3/945)*lambda_ - (5/9072)*(lambda_**2))
        dDelta_dx = (term1 - term2 - term3 + term4 + term5)/ (term6 + eps)
        return dDelta_dx
    
    def calc_del(self, D0, x):
        eps = 1e-9
        D0 = D0 + eps   # passing zero 
        del_ = odeint(self.dD_dx, D0, x)
        return del_
    
    def calc_vel_profile(self, del_, eta):
        self.F = 2*eta - 2*(eta**3) + eta**4
        self.G = (1/6)*eta*((1-eta)**3)
        lambda_ = (del_**2 / self.nu)*self.dUe_dx
        u = self.Ue*(self.F + lambda_*self.G)
        return u

    def calc_bl_props(self, del_):
        # get del_, del_star, theta, shape_factor(H) from self.h, self.g, lambda, Z (solved by ode)
        # del_ = self.calc_del()
        lambda_ = (del_**2 / self.nu)*self.dUe_dx
        del_star = del_*((3/10)  - (1/120)*lambda_)
        theta = del_*((37/315) - (1/945)*lambda_ - (1/9072)*(lambda_**2))
        tao = (self.mu*self.Ue / del_)*(2 + (1/6)*lambda_)
        return del_star, theta, tao



class Thwaites:

    def __init__(self):
        self.Ue, self.dUe_dx, self.d2Ue_dx2 = self.potential_flow_soln()

    def potential_flow_soln(pressure_grad, V_inf, rho):
        # supposed to solve potential flow in the region outside the BL
        dp_dx = pressure_grad
        Ue = V_inf
        dUe_dx = - dp_dx / (rho*Ue)
        d2Ue_dx2 = 0
        return Ue, dUe_dx, d2Ue_dx2
    

# x = np.linspace(0, 1, 100)
# eta = np.linspace(0, 6, 120)
# pol = Polhaussen(50 , 10, 1.225)
# D0 = 0.05
# del_ = pol.calc_del(D0, x)
# plt.plot(x, del_)
# plt.show()
# u_x0 = pol.calc_vel_profile(del_[20], eta)
# plt.plot(u_x0, eta)
# plt.show()