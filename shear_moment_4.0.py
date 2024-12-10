import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Load_Equations:

    def __init__(self, y_values, x_values):
        self.x_data = np.array(x_values)
        self.y_data = np.array(y_values)

        # Fit the data to the equation
        self.degree = 5
        self.params, self.covariance = curve_fit(lambda x, *p: self.polynomial(x, *p), self.x_data, self.y_data,
                                                 p0=[1] * (self.degree + 1))

        # Generate x values for the fitted curve
        self.x_fit = np.linspace(min(self.x_data), max(self.x_data), 500)
        self.y_fit = self.polynomial(self.x_fit, *self.params)

        self.plot_equation()

    def polynomial(self, x, *params):
        degree = len(params) - 1
        result = sum([params[i] * x ** (degree - i) for i in range(degree + 1)])
        return result

    def plot_equation(self):
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(self.x_data, self.y_data, color='blue', label='Design Load Data')
        plt.plot(self.x_fit, self.y_fit, color='red')
        plt.title('Design Load Curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()

        x = np.linspace(0, 10.92, 1000)
        coeffs = self.params
        y = np.polyval(coeffs, x)

        plt.subplot(1, 2, 2)
        plt.plot(x, y, label="Spanwise Load", color="b", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("Load Value")
        plt.title("Fitted Curve")
        plt.xlim(0, 10.92)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def get_equation(self):
        return '+'.join([(str(round(self.params[i], 4)) + "*x^" + str(len(self.params) - i - 1)) for i in
                         range(len(self.params))]).replace("+-", "-")

params = [0.031146807213254215, -0.9734557304800973, 11.46410433320769, -59.679379592049216, 95.23366768716704, 366.2564395680214, -1299.414656348635, -13516.355300199872]
def lift_force(y):
    return (sum([params[i] * y**(len(params)-1-i) for i in range(len(params))]))




def weight_force(y):
    return (4652.7156 - 289.69*y)

# Ask user for inputs
wing_length = 10.92
num_point_forces = int(input("Enter the number of point shear forces: "))
point_forces = []

for i in range(num_point_forces):
    location = float(input(f"Enter the location of point force {i+1} along the wing (in meters): "))
    magnitude = -float(input(f"Enter the magnitude of point force {i+1} (positive for downward, negative for upward): "))
    point_forces.append((location, magnitude))

# Set up the y-axis (spanwise direction)
y = np.linspace(0, wing_length, 1000)

# Calculate distributed forces
lift = lift_force(y)
weight = weight_force(y)
distributed_force = lift - weight  # Net distributed force (z direction)


# Calculate total distributed load and moment due to distributed forces
dy = y[1] - y[0]  # Spanwise step
total_distributed_load = np.sum(distributed_force * dy)
total_distributed_moment = np.sum(distributed_force * y * dy)


#test integral for lift
#lift_test = np.sum(lift*dy)
#print(f"the total lift is {lift_test}")

# Add contributions from point forces
total_point_load = sum(magnitude for _, magnitude in point_forces)
total_point_moment = sum(location * magnitude for location, magnitude in point_forces)

# Calculate reaction forces at root (y=0)
reaction_force = -(total_distributed_load + total_point_load)  # Vertical force balance
reaction_moment = (total_distributed_moment + total_point_moment)  # Moment balance about root
#print(f"The reaction force is {reaction_force} and the reaction moment is {reaction_moment}")

# Initialize arrays for shear force and bending moment
shear_force = np.zeros_like(y)
bending_moment = np.zeros_like(y)

# Start with reaction forces
shear_force[0] = reaction_force
bending_moment[0] = reaction_moment

# Compute shear force and moment along the wing span
for i in range(1, len(y)):
    shear_force[i] = shear_force[i-1] + distributed_force[i-1] * dy
    bending_moment[i] = bending_moment[i-1] + shear_force[i-1] * dy

# Add point forces to shear force and bending moment
for location, magnitude in point_forces:
    idx = np.searchsorted(y, location)
    shear_force[idx:] += magnitude
    for j in range(idx, len(y)):
        bending_moment[j] += magnitude * (y[j] - location)


#print(shear_force[0])
#print(bending_moment[0])


#uncomment for moment curve equation
moment_curve_fit = Load_Equations(bending_moment,y)
print(moment_curve_fit.params)
print(f"The equation for the moment curve is: {moment_curve_fit.get_equation()}")


# Plot the diagrams
plt.figure(figsize=(12, 8))

# Plot distributed force
#plt.subplot(3, 1, 1)
#plt.plot(y, distributed_force, label="Distributed Force (Lift - Weight)")
#plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
#plt.title("Distributed Force along Wing Span")
#plt.xlabel("Spanwise Location y (m)")
#plt.ylabel("Force (N/m)")
#plt.legend()
#plt.grid()

# Plot shear force
plt.subplot(2, 1, 1)
plt.plot(y, shear_force, label="Shear Force", color='orange')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("Shear Force Diagram")
plt.xlabel("Spanwise Location y (m)")
plt.ylabel("Shear Force (N)")
plt.legend()
plt.grid()

# Plot bending moment
plt.subplot(2, 1, 2)
plt.plot(y, bending_moment, label="Bending Moment", color='green')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("Bending Moment Diagram")
plt.xlabel("Spanwise Location y (m)")
plt.ylabel("Bending Moment (N·m)")
plt.legend()
plt.grid()


plt.tight_layout()
plt.savefig('shear_moment_graph')
plt.show()



#print(bending_moment)
#print("*"*100)
#print(y)

np.savetxt("bending_moment_values",bending_moment, fmt = "%e")
np.savetxt("x_values",y, fmt = "%e")

#print(y)
#print("done")
