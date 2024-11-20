import numpy as np
from scipy.integrate import quad

# Function w(a, b, c, y)
def w(y, a, b, c):
    return a * y**2 + b * y + c

# Step function
def u_y1(y, y1):
    return 1 if y >= y1 else 0

# Function of S(y)
def s(y, a, b, c, P1,P2, L, y1,y2):
    # Integral of w from y to L
    integral, _ = quad(w, y, L, args=(a, b, c))
    # Calculate S(y)
    return -(integral - P1 * (1 - u_y1(y, y1))-P2*(1-u_y1(y,y2)))

