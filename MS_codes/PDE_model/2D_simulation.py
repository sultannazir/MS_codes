import functions as fn
import numpy as np
import matplotlib.pyplot as plt

## MODEL PARAMETERS

Parameters = {
    # Cell
    'mu1' : 1.2/3600,        # max growth rate of strain 1 (in 1/sec)
    'mu2' : 1.32/3600,       # max growth rate of strain 2 (in 1/sec)
    'Cmax' : 0.05,           # max cell density (in g/mL dcw)
    'Y' : 0.5,               # biomass yield (in g biomass/g glucose)

    # Glucose
    'Ks' : 4*10**-6,         # rate of nutrient consumption by Monod kinetics (in g/mL)

    # Antibiotics
    'Km1' : 18*10**-6,       # Michaelis-Menton rate constant for antibiotic 1 inactivation by C1 (in g/mL)
    'Vmax1' : 4*10**2/3600,  # Maximum antibiotic 1 inactivation (by C1) rate (in mL/(g.sec))
    'IC1' : 10**-6,          # antibiotic 1 concentration at which C2 growth rate is halved (in g/mL)
    'Km2' : 18*10**-6,       # Michaelis-Menton rate constant for antibiotic 2 inactivation by C2 (in g/mL)
    'Vmax2' : 4*10**0/3600,  # Maximum antibiotic 2 inactivation (by C2) rate (in mL/(g.sec))
    'IC2' : 10**-6,          # antibiotic 2 concentration at which C1 growth rate is halved (in g/mL)

    # Diffusion
    'DC' : 100,              # diffusion coefficient of biomass (in um^2/sec)
    'DS' : 600,              # diffusion coefficient of glucose (in um^2/sec)
    'DA1' : 400,             # diffusion coefficient of antibiotic 1 (in um^2/sec)
    'DA2' : 400,             # diffusion coefficient of antibiotic 2 (in um^2/sec)

    # Initial condition
    'S0' : 0.04,            # glucose concentration in feed medium (in g/mL)
    'A10' : 0.04,           # antiobiotic 1 concentration in feed medium (in g/mL)
    'A20' : 0.04,           # antiobiotic 2 concentration in feed medium (in g/mL)
    'f1' : 0.1,              # initial density of loading sites for cooperators (fraction of grid points)
    'f2' : 0.1,              # initial density of loading sites for cheaters (fraction of grid points)
    'C0' : 0.01,             # inital biomass concentration at loading sites (dimensionless)

    # Frame width (x and y axis - assuming square frame) in um
    'Len' : 10000,
    # Space desolution in um
    'dx' : 100,
    # Time Resolution in sec
    'dt' : 1,

    # Time steps
    'T' : 200
}

fn.parameters_to_global_variables(Parameters)
keys = list(Parameters.keys())
for i in keys:
    globals()[i] = Parameters[i]

dx0 = dx/Len
dtau = dt*mu1
res = dtau/dx0**2
print("Frame dimensions are", Len, "X", Len, "micrometer sq. \n dx = ", dx, "micrometers")
print("Simulation time is", dt*T, "seconds \n dt =", dt, "seconds")
print("Dimensionless resolution, dtau / dx0^2 = ", res)

###

# DIMENSIONLESS PARAMETERS

Parameters_nondim = {
    'dC' : DC/(mu1*Len**2),            # diffusion constant of biomass
    'dS' : DS/(mu1*Len**2),            # diffusion constant of glucose
    'dA1' : DA1/(mu1*Len**2),          # diffusion constant of antibiotic 1
    'dA2' : DA2/(mu1*Len**2),          # diffusion constant of antibiotic 2

    'mu_r' : mu2/mu1,                 # relative instrinsic growth rate (of C2 wrt C1)

    'beta' : Cmax/(Y*Ks),             # Glucose utilization

    'alpha1' : Km1/IC1,               # Antibiotic 1 strength
    'alpha2' : Km2/IC2,               # Antibiotic 2 strength

    'gamma1' : Vmax1*Cmax/(mu1*IC1),  # Benefit from C1 to C2 by antibiotic 1 inactivation
    'gamma2' : Vmax2*Cmax/(mu1*IC2),  # Benefit from C2 to C1 by antibiotic 2 inactivation

    'dx0' : dx0,
    'dtau' : dtau,
    'res' : res
}

fn.parameters_to_global_variables(Parameters_nondim)

# 2D simulation

update_every = 15 # number of time steps after which data is stored
#C1_time = []
#C2_time = []
#S_time = []
#A1_time = []
#A2_time = []
C1, C2, S, A1, A2 = fn.initialize_2D()

for tt in range(T):
    if tt%update_every == 0:
        print(tt)
        #C1_time.append(C1.copy())
        #C2_time.append(C2.copy())
        #S_time.append(S.copy())
        #A1_time.append(A1.copy())
        #A2_time.append(A2.copy())
    C1, C2, S, A1, A2 = fn.update_2d(C1, C2, S, A1, A2)

print("end of simultaion")

fig = plt.figure(figsize=(8, 8), dpi=80)

biomass = C1 + C2
plt.subplot(221)
plt.imshow(np.divide(C1, biomass, out=np.zeros_like(C1), where=biomass != 0), vmin=0, vmax=1)
plt.title('Fraction of antiobiotic resistance')
plt.colorbar()

plt.subplot(222)
plt.imshow(biomass)
plt.title('Total biomass')

plt.subplot(223)
plt.imshow(A1)
plt.title('Antibiotic 1')

plt.subplot(224)
plt.imshow(A2)
plt.title('Antibiotic 2')

plt.tight_layout()
plt.show()

