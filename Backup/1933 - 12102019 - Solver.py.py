import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='mars', atmos_filename=None,
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2):
        """
            Set up the initial parameters and constants for the target planet

            Parameters
            ----------

            atmos_func : string, optional
                Function which computes atmospheric density, rho, at altitude, z.
                Default is the exponential function ``rho = rho0 exp(-z/H)``.
                Options are ``exponential``, ``tabular``, ``constant`` and ``mars``

            atmos_filename : string, optional
                If ``atmos_func`` = ``'tabular'``, then set the filename of the table
                to be read in here.

            Cd : float, optional
                The drag coefficient

            Ch : float, optional
                The heat transfer coefficient

            Q : float, optional
                The heat of ablation (J/kg)

            Cl : float, optional
                Lift coefficient

            alpha : float, optional
                Dispersion coefficient

            Rp : float, optional
                Planet radius (m)

            rho0 : float, optional
                Air density at zero altitude (kg/m^3)

            g : float, optional
                Surface gravity (m/s^2)

            H : float, optional
                Atmospheric scale height (m)

            Returns
            -------

            None
        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0

        if atmos_func == 'exponential':
            self.rhoa = lambda z: self.rho0 * np.exp(-z / self.H)
        elif atmos_func == 'tabular':
            raise NotImplementedError
        elif atmos_func == 'mars':
            self.rhoa = self.marsdensity(z)
        elif atmos_func == 'constant':
            self.rhoa = lambda z: rho0
        else:
            raise NotImplementedError

    def marsdensity(self, z):
        p = 0.699*np.exp(-0.00009*z)

        if z >= 7000:
            T = 249.7 - 0.00222*z
        else:
            T = 242.1 - 0.000998*z

        marsdensity = p / (0.1921*T)
        return marsdensity 
    def deg_to_rad(self, deg):
        """
            Returns an angle in radians
            for a given angle in degrees
        """
        return deg * np.pi / 180

    def rad_to_degrees(self, rad):
        """
            Returns an angle in degrees
            for a given angle in radians
        """
        return rad*180 / np.pi

    def impact(self, radius, velocity, density, strength, angle,
               init_altitude=100e3, dt=0.05, radians=False):
        """
            Solve the system of differential equations for a given impact event.
            Also calculates the kinetic energy lost per unit altitude and
            analyses the result to determine the outcome of the impact.

            Parameters
            ----------

            radius : float
                The radius of the asteroid in meters

            velocity : float
                The entery speed of the asteroid in meters/second

            density : float
                The density of the asteroid in kg/m^3

            strength : float
                The strength of the asteroid (i.e., the ram pressure above which
                fragmentation and spreading occurs) in N/m^2 (Pa)

            angle : float
                The initial trajectory angle of the asteroid to the horizontal
                By default, input is in degrees. If 'radians' is set to True, the
                input should be in radians

            init_altitude : float, optional
                Initial altitude in m

            dt : float, optional
                The output timestep, in s

            radians : logical, optional
                Whether angles should be given in degrees or radians. Default=False
                Angles returned in the DataFrame will have the same units as the
                input

            Returns
            -------

            Result : DataFrame
                A pandas DataFrame containing the solution to the system.
                Includes the following columns:
                ``velocity``, ``mass``, ``angle``, ``altitude``,
                ``distance``, ``radius``, ``time``, ``dedz``

            outcome : Dict
                dictionary with details of airburst and/or cratering event.
                For an airburst, this will contain the following keys:
                ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.

                For a cratering event, this will contain the following keys:
                ``impact_time``, ``impact_mass``, ``impact_speed``.

                All events should also contain an entry with the key ``outcome``,
                which should contain one of the following strings:
                ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """
        result = self.solve_atmospheric_entry(radius, velocity, density, strength, angle)
        result = self.calculate_energy(result)
        result_2 = earth.analyse_outcome(result)
        return result, result_2

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False):
        """
            Solve the system of differential equations for a given impact scenario

            Parameters
            ----------

            radius : float
                The radius of the asteroid in meters

            velocity : float
                The entery speed of the asteroid in meters/second

            density : float
                The density of the asteroid in kg/m^3

            strength : float
                The strength of the asteroid (i.e., the ram pressure above which
                fragmentation and spreading occurs) in N/m^2 (Pa)

            angle : float
                The initial trajectory angle of the asteroid to the horizontal
                By default, input is in degrees. If 'radians' is set to True, the
                input should be in radians

            init_altitude : float, optional
                Initial altitude in m

            dt : float, optional
                The output timestep, in s

            radians : logical, optional
                Whether angles should be given in degrees or radians. Default=False
                Angles returned in the DataFrame will have the same units as the
                input

            Returns
            -------
            Result : DataFrame
                A pandas DataFrame containing the solution to the system.
                Includes the following columns:
                ``velocity``, ``mass``, ``angle``, ``altitude``,
                ``distance``, ``radius``, ``time``
        """
        # initial state in order : [0] v [1] m [2] theta [3] z [4] x [5] r 
        # distance for (t=0) is assumed to be 0
        if radians == False :
            initial_state = np.array([velocity, density*4/3*np.pi*10**3, self.deg_to_rad(angle), init_altitude, 0, radius])
        elif radians == True :
            initial_state = np.array([velocity, density*4/3*np.pi*10**3, angle, init_altitude, 0, radius])
        state, time = self.RK_45(self.f_com, initial_state, strength, density)     
        velocity = state[:,0]
        mass = state[:,1]
        angle = state[:,2]
        altitude = state[:,3]
        distance = state[:,4]
        radius = state[:,5]
        result = pd.DataFrame({'velocity': velocity,
                             'mass': mass,
                             'angle': angle,
                             'altitude': altitude,
                             'distance': distance,
                             'radius': radius,
                             'time': time})
        return result

    def calculate_energy(self, result):
        """
            Function to calculate the kinetic energy lost per unit altitude in
            kilotons TNT per km, for a given solution.

            Parameters
            ----------

            result : DataFrame
                A pandas DataFrame with columns for the velocity, mass, angle,
                altitude, horizontal distance and radius as a function of time

            Returns
            -------

            Result : DataFrame
                Returns the DataFrame with additional column ``dedz`` which is the
                kinetic energy lost per unit altitude
        """
        result = result.copy()
        result = result.to_numpy()
        dedz = np.array((0.5*result[1:,1]*(result[1:,0]**2) - 0.5*result[:-1,1]*(result[:-1,0]**2))/(result[1:,3]-result[:-1,3]))
        dedz = np.insert(dedz,0,0)
        velocity = result[:,0]
        mass = result[:,1]
        angle = result[:,2]
        altitude = result[:,3]
        distance = result[:,4]
        radius = result[:,5]
        time = result[:,6]
        result = pd.DataFrame({'velocity': velocity,
                             'mass': mass,
                             'angle': angle,
                             'altitude': altitude,
                             'distance': distance,
                             'radius': radius,
                             'time': time,
                             'dedz': dedz})
        return result        

    def analyse_outcome(self, result):
        """
        Inspect a prefound solution to calculate the impact and airburst stats

        Parameters
        ----------

        result : DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time

        Returns
        -------

        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.

            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.

            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """

        # Enter your code here to process the result DataFrame and
        # populate the outcome dictionary.
        dedz_of_burst = result.dedz.max()
        row_of_burst = result.dedz.idxmax()
        burst_altitude = result.altitude[row_of_burst]
        burst_total_ke_lost = result['dedz'].iloc[:row_of_burst+1].sum()

        impact_time = result.loc[result.altitude == 0]['time'].item()
        impact_mass = result.loc[result.altitude == 0]['mass'].item()
        impact_speed = result.loc[result.altitude == 0]['velocity'].item()

        outcome={}
        if burst_altitude>5000:
            outcome["outcome"] = "Airburst"
            outcome["burst_peak_dedz"] = dedz_of_burst
            outcome["burst_altitude"] = burst_altitude
            outcome["burst_total_ke_lost"] = burst_total_ke_lost
        elif (burst_altitude<5000 and burst_altitude>0):
            outcome["outcome"] = "Airburst and cratering"
        else:
            outcome["outcome"] = "Cratering"
            outcome["impact_time"] = impact_time
            outcome["impact_mass"] = impact_mass
            outcome["impact_speed"] = impact_speed

        return outcome

    def f_com(self, t, state, strength,density):
        '''
            Composite of derivative function of velocity, mass, angle, altitude, distance, radius (in order)
            Parameters
            ----------
            t : time step
            
            state : array or list
                composite of current condition of velocity, mass, angle, altitude, distance, radius (in order) for every time step calculation

            Returns
            -------------
            a Numpy array containing derivative result of velocity, mass, angle, altitude, distance, radius (in order)
        '''
        v, m, theta, z, x, r = state
        f = np.zeros_like(state)
        A = np.pi * r**2
        # derivative function in order : [0] dv [1] dm [2] dtheta [3] dz [4] dx [5] dr
        f[0] = ((-self.Cd * self.rhoa(z)* A * v**2) / (2 * m)) + (self.g * np.sin(theta))
        f[1] = (-self.Ch * self.rhoa(z) * A * v**3) / (2* self.Q)
        f[2] = (self.g * np.cos(theta) / v) - (self.Cl * self.rhoa(z) * A * v / (2 * m)) - (v * np.cos(theta) / (self.Rp + z))
        f[3] = -v * np.sin(theta)
        f[4] = (v * np.cos(theta)) / (1 + (z / self.Rp))
        f[5] = (np.sqrt((7/2 * self.alpha * self.rhoa(z) / density)) * v if self.rhoa(z)*v**2 >= strength else 0)
        return f

    def RK_45(self, f, initial_state, strength, density,  t0=0.0, t_max=2000, dt=0.05 ):
        '''
            Return velocity, mass, angle, altitude, distance, radius (in order) calculation for each time step using Runge_Kutta 4th order
            Please note this function will consider burst effect in dr/dt calculation
            Parameters
            ----------
            f : function
                composite of derivative function of velocity, mass, angle, altitude, distance, radius (in order)
            
            initial state : array or list
                composite of initial condition of velocity, mass, angle, altitude, distance, radius (in order)

            t0 = float
                initial time

            t_max = float
                End of ODE calculation
            
            dt = float, optional
                Output time step, in s

            Returns
            -------------
            a Numpy array containing velocity, mass, angle, altitude, distance, radius (in order)
        '''
        u = np.array(initial_state)
        U_all = [u]
        t = np.array(t0)
        t_all = [t0]           
        while t < t_max:
            k1 = dt*f(t, u, strength, density)
            k2 = dt*f(t + 0.5*dt, u + 0.5*k1, strength, density)
            k3 = dt*f(t + 0.5*dt, u + 0.5*k2, strength, density)
            k4 = dt*f(t + dt, u + k3, strength, density)
            u = u + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
            U_all.append(u)
            t = t + dt
            t_all.append(t)
            if u[3] < 0:
                break
            if u[1] < 0:
                break    
        return np.array(U_all), np.array(t_all)



earth = Planet()
# initial state in order : [0] v [1] m [2] theta [3] z [4] x [5] r
# density = 3000
# initial_state = np.array([21000, density*4/3*np.pi*10**3, earth.deg_to_rad(45), 100e3, 0, 10])
# t0 = 0
# t_max = 100
# dt = 0.05
# Y = 10e5

result = earth.solve_atmospheric_entry(10, 21000, 3000, 10e8, 45)
print(result)
# result_2 = earth.calculate_energy(result)
# print(result_2)
# print(earth.analyse_outcome(result_2))
# ax = plt.gca()
# result_2.plot(kind='line', x='dedz', y='altitude', ax=ax)
# result = earth.impact(10, 21000, 3000, 10e5, 45)
# print(result)