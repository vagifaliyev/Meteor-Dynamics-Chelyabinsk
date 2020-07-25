import scipy.interpolate as si
import numpy as np
import pandas as pd
import armageddon.solver as sl
import matplotlib.pyplot as plt

def calculate_mse(radius=10, strength=1e5, method=False):
    '''
    calculate the error of numerical model to the data of Russia
    method 1 :finding the norm error of all the points
    method 2 :finding the error between the peak points
    
    Parameters
    ----------
    
    radius : float
        asteroid radius
    
    strength : float
        asteroid strength
        
    method : True (method 1)or False(method 2)
    
    Returns
    -------
    mse: float
        square distance between burst points of fit model and raw data
    '''
    limit = 1
    
    earth = sl.Planet()
    out = earth.solve_atmospheric_entry(
        radius=radius, angle=18.3, strength=strength, velocity=19.2e3, density=3300)
    out = earth.calculate_energy(out)

    altitude = out.iloc[:,3]
    energy = out.iloc[:,-1]

    v_a = pd.DataFrame({'z':altitude, 'ek':energy})
    valid = v_a.loc[v_a.iloc[:,-1] > limit]
    
    if valid.size == 0:
        return np.inf

    com_z = valid.iloc[:,0].to_numpy()/1e3
    com_ek = valid.iloc[:,1].to_numpy()
    com_DF = pd.DataFrame({'com_z':com_z, 'com_ek':com_ek})

    basic_DF = pd.read_csv('./data/ChelyabinskEnergyAltitude.csv')
    basic_DF = basic_DF.loc[basic_DF.iloc[:,1]>limit]
    basic_z, basic_ek = basic_DF.iloc[:,0].to_numpy(), basic_DF.iloc[:,1].to_numpy()
    
    basic_burst_ek = np.max(basic_ek)
    

    upper = np.max(basic_z)
    lower = np.min(basic_z)

    # add the points in fit model that are blank between range of altitude in original model
    if np.max(com_z) < upper:
        add_max = upper
        add_min = np.max(com_z)
        number = int(add_max - add_min)+1
        zeros = np.zeros(number)
        add_z = np.linspace(add_max-0.05, add_min+0.05, number)

        add_DF = pd.DataFrame({'com_z':add_z, 'com_ek':zeros})
        com_DF = add_DF.append(com_DF, ignore_index=True, sort=False)

    if np.min(com_z) > lower:
        add_max = np.min(com_z)
        add_min = lower
        number = int(add_max - add_min)+1
        zeros = np.zeros(number)
        add_z = np.linspace(add_min+0.05, add_max-0.05, number)

        add_DF = pd.DataFrame({'com_z':add_z, 'com_ek':zeros})
        com_DF = com_DF.append(add_DF, ignore_index=True, sort=False)

    # slice the points that are outside the range of altitude in original model
    # which is used for later interpolation
    com_DF = com_DF.loc[com_DF.iloc[:,0] < upper]
    com_DF = com_DF.loc[com_DF.iloc[:,0] > lower]
    lp = si.interp1d(basic_z, basic_ek)
    
    
    if method == False:
        com_z = com_DF.iloc[:,0].to_numpy()
        com_ek = com_DF.iloc[:,1].to_numpy()
        interpolate_ek = lp(com_z)
        com_burst_ek = com_ek.max()
        com_burst_z = com_z[np.where(com_ek==com_burst_ek)]
        basic_burst_z = basic_z[np.where(basic_ek==basic_burst_ek)]
        distance_burst = (com_burst_z - basic_burst_z)**2 + (com_burst_ek-basic_burst_ek)**2
        mse = distance_burst
    else:
        com_z = com_DF.iloc[:,0]
        com_ek = com_DF.iloc[:,1]
        interpolate_ek = lp(com_z)
        diff_ek = com_ek - interpolate_ek
        mse = np.linalg.norm(diff_ek)/np.sqrt(diff_ek.shape)
    

    return mse