import numpy as np

area_per_stringer = 1e-4
density = 3000


def b(y):     # height of both spars
    return 0.460 - 0.028676 * y


def d(y):       # distance between spars
    return 1.8045 - 0.1107 * y


def stringer_contribution(y, n):    #contribution to moment of inertia
    return ((b(y)/2) ** 2) * area_per_stringer * n


def moi_distribution(y, n, t):
    return ((d(y) * t * ((b(y)/2) ** 2))*2) + (((b(y) ** 3) * t/12)*2 + stringer_contribution(y, n))


def wingbox_area(y):  #enclosed area
    A = (d(y) * b(y))
    return A


def circumference(y):
    C = (b(y) + d(y)) * 2
    return C


def wingboxmass(y, t):
    return ((circumference(y) * t) * density)


def stringer_contribution2(y, n):   #contribution to polar moment of inertia
    return (area_per_stringer * n * (d(y) ** 2))/12 + (area_per_stringer * n * (b(y)) ** 2)


def polarmoi_distribution(y, n, t):
    return (4 * wingbox_area(y)**2 * t)/(circumference(y)) + stringer_contribution2(y, n)

