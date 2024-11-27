import math
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit  

#flight conditions (cruise) 
V=285.5
rho=0.3796
n=2.5

q=0.5*rho*V**2
cr=4.03
ct=1.29
b=21.82
S=58.068
W=25256 

class Loads:

    def __init__(self, filename):        
        self.ylist = []
        self.Cllst = []
        self.Cdlst = []
        self.Cmlst = []     
        self.read=False
        self.file=filename  

        self.open_file()
        
        self.y_Cl_inter = sp.interpolate.interp1d(self.ylist, self.Cllst, kind = 'linear', fill_value= "extrapolate")
        self.y_Cd_inter = sp.interpolate.interp1d(self.ylist, self.Cdlst, kind = 'linear', fill_value= "extrapolate")
        self.y_Cm_inter = sp.interpolate.interp1d(self.ylist, self.Cmlst, kind = 'linear', fill_value= "extrapolate")

        print(len(self.ylist))

    def open_file(self):
        with open(self.file) as file:
            for line in file:
                try:
                    line=line.split()
                    if line[0]=="CL":
                        self.CL=float(line[2])
                    if line[0]=="y-span":
                        self.read=True
                        continue
                    if self.read==True:
                        if len(self.ylist)>1:
                            if self.ylist[-1]==float(line[0]) and self.ylist[-1]!=[]:
                                break
                        self.ylist.append(float(line[0][1:]))
                        self.Cllst.append(float(line[3]))
                        self.Cdlst.append(float(line[5]))
                        self.Cmlst.append(float(line[7]))
                except IndexError:
                    continue

class Load_Equations:

    def __init__(self, data):
        self.x_data=np.array([data[i][2] for i in range(len(data))])
        self.y_data=np.array([data[i][0] for i in range(len(data))])

        #Fit the data to the equation
        self.degree = 12
        self.params, self.covariance = curve_fit(lambda x, *p: self.polynomial(x, *p), self.x_data, self.y_data, p0=[1]*(self.degree + 1))

        #Generate x values for the fitted curve
        self.x_fit = np.linspace(min(self.x_data), max(self.x_data), 500)
        self.y_fit = self.polynomial(self.x_fit, *self.params)

        self.plot_equation()   

    def polynomial(self, x, *params):
        degree = len(params) - 1
        result = sum([params[i] * x**(degree - i) for i in range(degree + 1)])
        return result

    def plot_equation(self):
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1,2,1)
        plt.scatter(self.x_data, self.y_data, color='blue', label='desired Load Data')
        plt.plot(self.x_fit, self.y_fit, color='red')
        plt.title('desired Load Curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()

        x = np.linspace(0, 10.91, 500)
        coeffs = self.params
        y = np.polyval(coeffs, x)
        
        plt.subplot(1,2,2)
        plt.plot(x,y, label="Spanwise Load", color="b", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("Load Value")
        plt.title("Fitted Curve")
        plt.xlim(0, 10.91)  
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def get_equation(self):
        return '+'.join([(str(round(self.params[i],4))+"*x^"+str(len(self.params)-i-1)) for i in range(len(self.params))]).replace("+-","-")

a0=Loads(r"MainWing_a0.txt")
a10=Loads(r"MainWing_a10.txt")

Cl_d=2*n*W*9.80665/(V**2*rho*S)*1.1 
Cl_0 =  a0.CL       
Cl_10 =  a10.CL     
K = ((Cl_d - Cl_0)/(Cl_10 - Cl_0))

def desired_lift(y):
    return a0.y_Cl_inter(y) + K * (a10.y_Cl_inter(y) - a0.y_Cl_inter(y))

def desired_drag(y):
    return a0.y_Cd_inter(y) + K**2 * (a10.y_Cd_inter(y) - a0.y_Cd_inter(y))

def desired_moment(y):
    return a0.y_Cm_inter(y) + K * (a10.y_Cm_inter(y) - a0.y_Cm_inter(y))

def desired_AOA():
    return ((Cl_d-Cl_0)/(Cl_10-Cl_0))*10

print(desired_AOA())

spanwise_desired_lift = [((float(desired_lift(y))*math.cos(desired_AOA()*math.pi/180)+float(desired_drag(y))*math.sin(desired_AOA()*math.pi/180))*q*(cr-((cr-ct)*2/b)*float(y)),(cr-((cr-ct)*2/b)*float(y)),float(y)) for y in np.arange(0,10.91,0.2)]
spanwise_desired_drag = [((-float(desired_lift(y))*math.sin(desired_AOA()*math.pi/180)+float(desired_drag(y))*math.cos(desired_AOA()*math.pi/180))*q*(cr-((cr-ct)*2/b)*float(y)),(cr-((cr-ct)*2/b)*float(y)),float(y)) for y in np.arange(0,10.91,0.2)]
spanwise_desired_moment = [(float(desired_moment(y))*q*(cr-((cr-ct)*2/b)*float(y)),(cr-((cr-ct)*2/b)*float(y)),float(y)) for y in np.arange(0,10.91,0.2)] #moment about quarter chord is not affected by angle of attack 

a_d = desired_AOA()
print("desired angle of attack")
print(a_d)

spanwise_lift=Load_Equations(spanwise_desired_lift)
print("spanwise_lift")
print([float(i) for i in spanwise_lift.params])
print()

spanwise_drag=Load_Equations(spanwise_desired_drag)
print("spanwise_drag")
print(spanwise_drag.params)
print()

spanwise_moment=Load_Equations(spanwise_desired_moment)
print("spanwise_moment")
print(spanwise_moment.params)

