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
    return 0 if x > b_eng else 1

def q(x):
    return 0.0311*x**7-0.9735*x**6+11.4641*x**5-59.6794*x**4+95.2337*x**3+366.2564*x**2-1299.4147*x-13516.3553

def qd(x):
    load_factor_lift = 1
    return d(x) * q(x) * load_factor_lift

# Input correct values
thrust = 64499 # Thrust force in N
fan_diameter = 1.17
thickness_flexural_axis = 0.30
span = 10.98  # Total span (m)
b_eng = 3.87  # Position of the engine along the span (m)
load_factor = 1  # Acceleration dependent
W_eng = 1111.3 * 9.81 * load_factor  # Engine weight (N)
c_eng = 0.6615  # Factor for engine vertical torque contribution
h_engine = (fan_diameter / 2 + thickness_flexural_axis) / 2.0

# Torsion calculation function
def torsion():
    torsion_lst = []
    x_lst = np.arange(0, span, 0.01)  # Create x values with a step of 0.01

    for x in x_lst:
        shear_accum, _ = scipy.integrate.quad(qd, x, span)

        torque = shear_accum
        torsion_lst.append(torque)

    # Fit a polynomial to the torsion distribution
    polynomial_degree = 5  # Degree of the polynomial
    coefficients = np.polyfit(x_lst, torsion_lst, polynomial_degree)
    polynomial = np.poly1d(coefficients)

    # Generate polynomial values
    torsion_fit = polynomial(x_lst)

    # Plot torsion distribution and polynomial fit
    plt.plot(x_lst, torsion_lst, label="Torque Distribution")
    plt.plot(x_lst, torsion_fit, label=f"Polynomial Fit (degree {polynomial_degree})", linestyle='--')
    plt.title("Torque Distribution and Polynomial Fit")
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Span Position (m)")
    plt.grid()
    plt.legend()
    plt.show()

    print("Polynomial Coefficients:")
    print(polynomial)

    return torsion_lst, x_lst, polynomial

# Call the function to calculate and plot torsion
torsion_values, span_positions, polynomial_function = torsion()

# Save results
np.savetxt("torsion_values.txt", torsion_values, fmt="%e")
np.savetxt("span_positions.txt", span_positions, fmt="%e")
