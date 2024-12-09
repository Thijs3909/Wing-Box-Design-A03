import scipy as sp
from scipy.integrate import quad
import math
import numpy as np

from geometry_42 import moi_distribution
from geometry_42 import polarmoi_distribution
from geometry_42 import wingboxmass

# Import Function

#from loading_functions_41 import torque_distribution

def moment_distribution(y):
    return -10.0332*y**5 + 273.5632*y**4 - 2785.227*y**3 + 26071.1829*y**2 - 290785.6037*y**1 + 1361285.416
#the function below is the torque distribution for n= 2.5 load factor
def torque_distribution(y):
    if y<3.87 :
        return 0.1609*y**5 - 5.418*y**4 + 64.46*y**3 + 443.9*y**2 - 17970*y + 111400
    else:
        return 0.1609*y**5 - 5.418*y**4 + 64.46*y**3 + 443.9*y**2 - 17970*y + 111400 + 25917.03 - 7211.58
#The function below is the torque distrubution function for n = -1 load factor
#def torque_distribution(y):
    if y<3.87 :
        return -0.1366*y**5 + 4.388*y**5 - 52.88*y**3 - 4.375*y**2 + 6554*y - 43380
    else: 
        return -0.1366*y**5 + 4.388*y**5 - 52.88*y**3 - 4.375*y**2 + 6554*y - 43380 + 25917.03 - 7211.58


# Parameters

E = 71e9
G = 27e9
rho = 3000
b = 21.84
m_stringer = 50e-6 * rho * b/2

# Number of control points
y_points = [b/2]

# Displacement Integration

def integrand(y, n, t):
    return - (moment_distribution(y) / (E * moi_distribution(y, n, t)))

def angle(y, n, t):
    result, angleerror = quad(integrand, 0, y, args=(n, t))
    return result

def displacement(y, n, t):
    result, displacementerror = quad(angle, 0, y, args=(n, t))
    return result

# Twist Angle

def torqueintegrand(y, n, t):
    return - (torque_distribution(y)) / (G * polarmoi_distribution(y, n, t))

def twistangle(y, n, t):
    result, angleerror = quad(torqueintegrand, 0, y, args=(n, t))
    return result

# Optimisation

displacement_distribution_list = []
twist_distribution_list = []
configuration_mass_list = []
n_list = []
t_list = []

counter = 0

for n in range(0,4,2):
    for t in np.linspace(1.25e-3,2.5e-3, 10):
        displacementcontrolpoints = [displacement(y, n, t) for y in y_points]
        twistcontrolpoints = [twistangle(y, n, t) for y in y_points]

        #Calculate mass for configuration

        configuration_mass = quad(wingboxmass, 0, b/2, args=(t))[0] + n*m_stringer

        # Update lists

        displacement_distribution_list.append(displacementcontrolpoints)
        twist_distribution_list.append(twistcontrolpoints)
        configuration_mass_list.append(configuration_mass)
        n_list.append(n)
        t_list.append(t)
        counter +=1
        print(counter)

## Select design that meet requirements and save its mass

check_displacement = np.array(displacement_distribution_list)
check_twist = np.array(twist_distribution_list)
check_mass = np.array(configuration_mass_list)
check_n = np.array(n_list)
check_t = np.array(t_list)

displacementmask = np.all(np.abs(check_displacement) <= b*0.15, 1)
twistmask = np.all(np.abs(check_twist) <= 10, 1)

combined_mask = displacementmask & twistmask

sufficient_displacement_distribution_list = check_displacement[combined_mask]
sufficient_twist_distribution_list = check_twist[combined_mask]
sufficient_configuration_mass_list = check_mass[combined_mask]
sufficient_n_list = check_n[combined_mask]
sufficient_t_list = check_t[combined_mask]

## Filter for the design with the lowest mass

index_best_design = np.argmin(sufficient_configuration_mass_list)

print(sufficient_displacement_distribution_list[index_best_design], sufficient_twist_distribution_list[index_best_design], sufficient_configuration_mass_list[index_best_design])
print(sufficient_n_list[index_best_design], sufficient_t_list[index_best_design])
