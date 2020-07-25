import pandas as pd
import numpy as np
#from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
import dask
from dask import delayed

def solve_ensemble(planet, fiducial_impact, variables, radians=False, rmin = 8, rmax =12 ):
    samples = 10
    nval = int(11) #for bins
    Y_min, Y_max = 1e3, 10e6
    p_m, stress_p = 3000, 1000
    #probabilities = np.linspace(0,1,samples)
    #P = np.linspace(0.00001, 0.99999, samples)
    
    radius = np.full((samples,),fiducial_impact['radius'])
    angle = np.full((samples,),fiducial_impact['angle'])
    strength = np.full((samples,),fiducial_impact['strength'])
    velocity = np.full((samples,),fiducial_impact['velocity'])
    density = np.full((samples,),fiducial_impact['density'])
    
    # var_rad = probabilities * (rmax - rmin) + rmin
    # var_ang = np.arccos(np.sqrt(1-probabilities))
    # var_ang[var_ang<1] = 1 #not allowing 0 deg 
    # var_str = 10** (P * (np.log10(Y_max/Y_min)) + np.log10(Y_min))
    # var_den = erfinv(P*2-1)*stress_p*np.sqrt(2)+p_m

    columns = []

    #finding probility distribution. using diff to convert to 
    for var in variables:
        if var == 'radius':
            #uniform distribution
            radius = np.random.uniform(rmin, rmax, samples)
            columns.append(radius)
        if var == 'angle':
            angles = np.linspace(0,90,nval)
            angles_dist = 2*np.sin(np.radians(angles))*np.cos(np.radians(angles))
            angles_dist = angles_dist/np.sum(angles_dist) # normalizing to add up to 1
            angless = np.random.choice(angles, size=samples, p=angles_dist)
            columns.append(angless)
        if var == 'strength':
            str = np.linspace(Y_min, Y_max,nval)
            str_dist = 1/(str*4)
            str_dist = str_dist/np.sum(str_dist)
            strengths = np.random.choice(str, size=samples, p=str_dist)
            columns.append(strengths)
        if var == 'velocity':
            # nf_velocity = np.array([inverse_F(u,11) for u in probabilities])
            # v_escape = 11.2
            # velocity = np.sqrt(v_escape**2 +inf_velocity**2)*1e3
            vlc = np.linspace(0,50000,nval) # infinite distance
            vlci = np.sqrt(11200**2 + vlc**2) # impact velocity
            vlc_dist = (np.sqrt(2/np.pi)*np.exp(-(vlc/1000)**2/242)*(vlc/1000)**2)/1331
            vlc_dist = vlc_dist/np.sum(vlc_dist)
            velocitys = np.random.choice(vlci, size=samples, p=vlc_dist)
            columns.append(velocitys)
        if var == 'density':
            den = np.linspace(1,7001,nval)
            den_dist = np.exp(-(den-p_m)**2/2e6)/(stress_p*np.sqrt(2*np.pi))
            den_dist = den_dist/np.sum(den_dist)
            densitys = np.random.choice(den,size=samples, p=den_dist)
            columns.append(densitys)

    # Ensemble function
    outcomes = []

    dask.config.set(scheduler = 'processes')
    for i in range(samples):
        output, outcome= delayed(planet.impact, nout=2)(radius=radius[i], 
            angle=angle[i], strength=strength[i], velocity=velocity[i], density=density[i], init_altitude= 100000, dt = 0.5)
        outcomes.append(outcome)
    outcomes = dask.compute(*outcomes)
    return(pd.DataFrame.from_dict(outcomes))

# def F(x, a):
#     return  erf(x/(a*np.sqrt(2)))-(x/a)*np.exp(-x**2/(2*a**2))*np.sqrt(2/np.pi)
# def inverse_F(p, a):
#     candidates = np.linspace(0, 500, 10000)
#     for x in candidates:
#         if F(x, a) >= p:
#             return x 
#     return 500