import numpy as np
import scipy
import sympy as sm
from scipy import integrate
import matplotlib.pyplot as plt

# Functions
def u_1(x):
    return np.heaviside(x, 1)

def u_2(x):
    return np.heaviside(x, 2)

def d(x):
    c = 4.01 - 0.2495 * x
    return 0.1185 * c

def T_inf(x):
    # Returns 0 or 1 based on the x position relative to b_eng
    return 0 if x > b_eng else 1

def q(x):
    return -0.12*x**7+4.1617*x**6-57.363*x**5+399.4055*x**4-1470.8353*x**3+2536.1491*x**2-2163.9095*x+38619.7891
  # Fixed exponentiation from ^ to **

def qd(x):
    load_factor_lift = 1
    return d(x) * q(x) * load_factor_lift

# Input correct values
thrust = 64499 # Thrust force in N
fan_diameter = 1.17
thickness_flexural_axis = 0.30
span = 10.98  # Total span (m)
b_eng = 3.87  # Position of the engine along the span (m)
load_factor = 1 #acceleration dependant
W_eng = 1111.3*9.81*load_factor  # Engine weight (N)
c_eng = 0.6615 # Factor for engine vertical torque contribution
h_engine = (fan_diameter/2+thickness_flexural_axis)/2.0
# Torsion calculation function

def torsion():
    torsion_lst = []
    x_lst = np.arange(0, span, 0.01)  # Create x values with a step of 0.01

    for x in x_lst:

        shear_accum, _ = scipy.integrate.quad(qd, x, span)
        thrust_point_force = T_inf(x) * thrust * np.cos(np.radians(24.5)) * h_engine
        engine_W_force = T_inf(x) * W_eng * c_eng
        torque = shear_accum + engine_W_force - thrust_point_force

        print(
            f"x={x:.2f}, shear_accum={shear_accum:.2f}, thrust_point_force={thrust_point_force:.2f}, engine_W_force={engine_W_force:.2f}, torque={torque:.2f}")

        torsion_lst.append(torque)

    # Plot torsion distribution
    plt.figure(figsize=(30,7))
    plt.plot(x_lst, torsion_lst, linewidth=4)
    plt.title("Torque Distribution", fontsize=25)
    plt.ylabel("Torque (Nm)",fontsize = 25)
    plt.xlabel("Spanwise location(y) (m)", fontsize = 25)
    plt.tick_params(axis='both', labelsize=20)  # Tick label font size
    plt.legend(fontsize=20)  # Legend font size
    plt.grid()
    plt.show()

    return torsion_lst, x_lst

# Call the function to calculate and plot torsion
torsion_values, span_positions = torsion()

np.savetxt("torsion_values.txt", torsion_values, fmt = "%e")
np.savetxt("span_positions.txt", span_positions, fmt = "%e")
