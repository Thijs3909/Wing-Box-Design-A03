import matplotlib.pyplot as plt
import numpy as np
import scipy as scp


altitude = 0
#altitude=10668
OEW=14707 #kg
MTOW = 25256 #kg
MTOW=OEW+9302
wingSurface = 58.04 #m^2


def ISAresults(alt):
    temp = 288.15 - 0.00649 * alt
    dens = 1.225 * (1-22.558*10**(-6)*alt)**4.2559
    Vsound = np.sqrt(1.4 * 287 * temp)
    return (temp, dens, Vsound)

ISAValues = ISAresults(altitude)

# https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/sound.html
# https://eng.libretexts.org/Bookshelves/Aerospace_Engineering/Fundamentals_of_Aerospace_Engineering_(Arnedo)/02%3A_Generalities/2.03%3A_Standard_atmosphere/2.3.03%3A_ISA_equations

Vc=cnstCruiseSpeed = 0.77*ISAValues[2]

#max manoeuvring
nmax = min(max(2.1 + 24000/(MTOW/0.454 + 10000), 2.5), 3.8)

#minimum manoeuvring
nmin = -1

#flaps deflected maximum load
nflaps = 2

#Stall Speeds
def speedCalc(CL, surface, weight):
    return np.sqrt((2*weight*9.80665)/(ISAValues[1]*surface*CL))

#VS0 for different flap configurations landing, takeoff and cruise
#!!!!!!ADD CRUISE MASS FRACTION and proper CLs
flapStallSpeed = {
    "landing": speedCalc(2.5,wingSurface+12.3,0.89*MTOW),
    "takeOff": speedCalc(2.35,wingSurface+3.31+2.84,MTOW)
}

Vs0=flapStallSpeed["takeOff"]

#Stall speed when flaps are completely retracted
Vs1 = speedCalc(1.096,wingSurface,MTOW) #this mass is incorrect

#Manouvering speed
manouverSpeed = Vs1 * np.sqrt(nmax)

#Design dive speed function
def designSpeedCalc(Vsound = 343):
    Vd1 = cnstCruiseSpeed / 0.8
    Mc = 0.77
    Md1 = (Mc / 0.8) *Vsound #converted to velocity for return function

    #Vd2 = 0.05 * Mc*Vsound + cnstCruiseSpeed
    #Md2 = (0.05 * Mc + Mc) * Vsound
    Vd2=Md2=1000000
    #the most critical case is the minimum velocity
    return min(Vd1, Vd2, Md1, Md2 )

Vd = designSpeedCalc(ISAValues[2])

# Design wing-flap speed What??
def wingFlapCalc():
    VF1 = 1.6 * Vs1
    VF2 = 1.8 * Vs1
    VF3 = 1.8 * flapStallSpeed["landing"]
    return max(VF1, VF2, VF3)

Vf = wingFlapCalc()

#p=np.sqrt(1.225/ISAValues[1])
#Vs0/=p
#Vs1/=p
#Vd/=p
#Vf/=p
#Vc/=p








nr=1000
x=np.linspace(0,Vd,nr)
#y1 is flapsup positive n portion
#y2 is flapsdown positive n portion
#y3 is flapsup/down negative n portion
#define point A
y1=(x/Vs1)**2
for i in range(0 , nr):
    if y1[i]>=nmax:
        y1[i]=nmax
y1[999]=0

y3=-(x/Vs1)**2
for i in range(0 , nr):
    if y3[i]<=nmin:
        y3[i]=nmin
for i in range(int(Vc/Vd*nr) , nr):
    y3[i]=nmin+nmin*((x[i]-Vc)/(Vc-Vd))

y2=np.where(x<=Vf,(x/Vs0)**2, np.nan)
for i in range(0 , nr):
    if y2[i]>=nflaps:
        y2[i]=nflaps

for i in range(0 , nr):
    if y2[i]<y1[i]:
        y2[i]=np.nan

for i in range(0,nr):
    if (x[i]>Vf):
        y2[i]=y1[i]
        break

Vm=Vs1*np.sqrt(nmax)
print("VS="+str(Vs1))




plt.plot(x, y1, label="flapsup", color="blue")
plt.plot(x, y2, label="flaps down", color="green")
plt.plot(x, y3, label="lower", color="blue")

# Step 4: Customize the plot
plt.title(r"$V_{TAS}-n$ diagram")
plt.xlabel(r"$V_{TAS}$")
plt.ylabel(r"$n$")
plt.xticks([Vs1,Vs1*np.sqrt(nmax), Vc , Vd], labels=[r"$V_S$",r"$V_A$", r"$V_C$", r"$V_D$"])
plt.yticks([-1,1,nmax], labels=["-1", "1",np.round(nmax,2)])

plt.annotate('Flaps down', xy=(Vs0, y2[int(Vs0/Vd*1000)]), xytext=(0,1.2),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.annotate('A', xy=(Vs1*np.sqrt(nmax), nmax), xytext=(Vs1*np.sqrt(nmax)-6, nmax))
plt.annotate('D', xy=(Vd, nmax), xytext=(Vd+2, nmax))
plt.annotate('H', xy=(Vs1*np.sqrt(abs(nmin)), nmin), xytext=(Vs1*np.sqrt(abs(nmin))-6, nmin-0.1))
plt.annotate('F', xy=(Vc, nmin), xytext=(Vc+2, nmin-0.1))
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # Add x-axis
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")  # Add y-axis
plt.grid(True)  # Add gridlines

# Step 5: Display the plot
plt.show()



#for i in range(0,int(Vs0*np.sqrt(nmax)/Vd))
 #   y1[i]






    