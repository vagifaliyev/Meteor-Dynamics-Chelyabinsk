import armageddon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

earth = armageddon.Planet()

fiducial_impact = {'radius': 10.0,
                   'angle': 45.0,
                   'strength': 100000.0,
                   'velocity': 21000.0,
                   'density': 3000.0}

print('Starting simulation now...')

sample_size = 2
for i in range(sample_size):
    start_time = time.time()
    ensemble = armageddon.ensemble.solve_ensemble(earth,
                                              fiducial_impact,
                                              variables=['angle', 'radius', 'strength', 'velocity', 'density'], radians=False,
                                              rmin=8, rmax=12)
    print("--- %s seconds ---" % (time.time() - start_time))
print(ensemble)
