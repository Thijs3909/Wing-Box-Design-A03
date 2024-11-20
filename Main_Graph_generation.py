import Shear_function as shear
import matplotlib.pyplot as plt
import numpy as np

y = 0

no_shear_forces = int(input("Enter the number of point shear forces: "))
shear_forces_list = [1]*no_shear_forces
print(shear_forces_list) #debug

location_point_shear_lst = [1]*len(shear_forces_list)
for i in range(len(shear_forces_list)):
    location_point_shear_lst[i] = float(input(f"Enter the location of point shear force {i+1} in meters from the root: "))

print(*location_point_shear_lst) #debug

for i in range(len(shear_forces_list)):
    shear_forces_list[i] = float(input(f"Enter point shear force {i+1}, respecting the sign convention defined in the group"))

print(*shear_forces_list)

length_wing = float(input("Enter the length of the wing(m): "))

#start code for shear:
print("This code assumes a quadratic load distribution of the lift")
a = float(input("Enter the coefficient of the x^2 term :"))
b = float(input("Enter the coefficient of the x term: "))
c = float(input("Enter the intercept term: "))

y_values = np.linspace(0,length_wing, 500)
Shear_values = [shear.s(y,a,b,c,shear_forces_list[0], shear_forces_list[1],length_wing,location_point_shear_lst[0],location_point_shear_lst[1]) for y in y_values]

plt.figure(figsize=(10, 6))
plt.plot(y_values, Shear_values, label='V(x)', color='blue')
plt.title("Graph of V(x)")
plt.xlabel("y (Position along the wing)")
plt.ylabel("S(x) (Shear Force)")
plt.grid(True)
plt.legend()
plt.show()







