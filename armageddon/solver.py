import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline
import os

dirname = os.path.dirname(__file__)
class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential', atmos_filename=os.path.join(dirname, '../data/AltitudeDensityTable.csv'),
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
        self.atmos_filename = atmos_filename
        self.tabular = {}
        
        if atmos_func == 'exponential':
            self.rhoa = lambda z: self.rho0 * np.exp(-z / self.H)
        elif atmos_func == 'tabular':
            # read in atmosphere csv 
            atmos = pd.read_csv(self.atmos_filename, skiprows=6, delimiter=' ', names = ['Altitude', 'Density', 'Height'])

            z_zero_rho = atmos.iloc[0].Density
            z_max = atmos.iloc[-1].Altitude
            z_max_rho = atmos.iloc[-1].Density

            for i in range(len(atmos)):
                self.tabular[int(atmos.iloc[i].Altitude)] = (atmos.iloc[i].Density, atmos.iloc[i].Height)

            self.rhoa = lambda z: z_zero_rho if int(z) <= 0 else self.tabular[int(int(z/10)*10)][0]*np.exp((int(int(z/10)*10)-int(z))/self.tabular[int(int(z/10)*10)][1]) if int(z) < z_max else z_max_rho            

        elif atmos_func == 'mars':
            self.rhoa = lambda z: self.marsdensity(z)
        elif atmos_func == 'constant':
            self.rhoa = lambda z: rho0
        else:
            raise NotImplementedError

    def marsdensity(self, z):
        '''
        Solve mars density for given height for everytime step

        Parameters
        ----------
        z = altitude (float) 

        Returns
        ---------
        mars density for given height
        '''
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
        return np.round(deg * np.pi / 180,5)

    def rad_to_degrees(self, rad):
        """
        Returns an angle in degrees
        for a given angle in radians
        """
        return np.round(rad*180 / np.pi,3)

    def impact(self, radius, velocity, density, strength, angle,
               init_altitude=100e3, dt=0.5, radians=False):
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
        outcome = self.analyse_outcome(result)
        return result, outcome

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle, method='RK45',
            init_altitude=100e3, dt=0.01, radians=False):
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
        
        method : Runge Kutta 4th Order ('RK45') or Forward Euler ('FE')

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
            angle *= np.pi/180

        initial_state = np.array([velocity, density*4/3*np.pi*radius**3, angle, init_altitude, 0, radius])
        
        if method == 'RK45':
            state, time = self.RK_45(self.f_com, initial_state, strength, density) 
        elif method == 'FE':
            state, time = self.forward_euler(self.f_com, initial_state, strength, density) 
        elif method == 'RFK45':
            state, time = self.RFK_45(self.f_com, initial_state, strength, density)   
        else :
            raise NameError('Please select between RK45 or Forward Euler')
        
        state[np.isnan(state)] = 0

        # fitting solution to fixed output time step
        eq_vel = CubicSpline(time,state[:,0])
        eq_mass = CubicSpline(time,state[:,1])
        eq_angle = CubicSpline(time,self.rad_to_degrees(state[:,2]))
        eq_alt = CubicSpline(time,state[:,3])
        eq_dis = CubicSpline(time,state[:,4])        
        eq_rad = CubicSpline(time,state[:,5])

        n_entries = int(time[-1]/dt) + 1
        time_arr = np.linspace(0,time[-1], n_entries)
        vel = eq_vel(time_arr)
        mass = eq_mass(time_arr)
        angle = eq_angle(time_arr)
        alt = eq_alt(time_arr)
        dis = eq_dis(time_arr)
        rad = eq_rad(time_arr)

        # result = pd.DataFrame({'velocity': state[:,0],
        #                      'mass': state[:,1],
        #                      'angle': self.rad_to_degrees(state[:,2]),
        #                      'altitude': state[:,3],
        #                      'distance': state[:,4],
        #                      'radius': state[:,5],
        #                      'time': time})
        result = pd.DataFrame({'velocity': vel,
                             'mass': mass,
                             'angle': angle,
                             'altitude': alt,
                             'distance': dis,
                             'radius': rad,
                             'time': time_arr})
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
        dedz = np.array((0.5*result[1:,1]*(result[1:,0]*result[1:,0]) 
            - 0.5*result[:-1,1]*(result[:-1,0]*result[:-1,0]))/(result[1:,3]-result[:-1,3]))
        dedz = np.insert(dedz,0,0)

        result = pd.DataFrame({'velocity': result[:,0],
                             'mass': result[:,1],
                             'angle': result[:,2],
                             'altitude': result[:,3],
                             'distance': result[:,4],
                             'radius': result[:,5],
                             'time': result[:,6],
                             'dedz': dedz / 4.184e9})
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
        dedz_of_burst = result.dedz.max()
        row_of_burst = result.dedz.idxmax()
        burst_altitude = result.altitude[row_of_burst]

        burst_total_ke_lost = (1/2*(result['mass'][0]*result['velocity'][0]*result['velocity'][0]
            -result['mass'][row_of_burst]*result['velocity'][row_of_burst]*result['velocity'][row_of_burst]))/ 4.184e12

        impact_time = result.at[row_of_burst,'time']
        impact_mass = result.at[row_of_burst,'mass']
        impact_speed = result.at[row_of_burst,'velocity']

        outcome={}
        if burst_altitude>5000:
            outcome["outcome"] = "Airburst"
            outcome["burst_peak_dedz"] = dedz_of_burst
            outcome["burst_altitude"] = burst_altitude
            outcome["burst_total_ke_lost"] = burst_total_ke_lost
        elif (burst_altitude<5000 and burst_altitude>0):
            outcome["outcome"] = "Airburst and cratering"
            outcome["burst_peak_dedz"] = dedz_of_burst
            outcome["burst_altitude"] = burst_altitude
            outcome["burst_total_ke_lost"] = burst_total_ke_lost           
        else:
            outcome["outcome"] = "Cratering"
            outcome["impact_time"] = impact_time
            outcome["impact_mass"] = impact_mass
            outcome["impact_speed"] = impact_speed

        return outcome

    def f_com(self, t, state, strength, density):
        '''
        Composite of derivative function of velocity, mass, angle, altitude, distance, radius (in order)
        Parameters
        ----------
        t : time step

        state : array or list
            composite of current condition of velocity, mass, angle, altitude, distance, radius (in order) for every time step calculation

        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)

        density : float
            The density of the asteroid in kg/m^3                

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


    def RK_45(self, f, initial_state, strength, density,  t0=0.0, t_max=20000, dt=0.1):
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

    def RFK_45(self, f, initial_state, strength, density,  t0=0.0, t_max=20000, dt=0.1):
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
        dt_min = 0.01
        dt_max = 0.5
        e_min = 1e2     
        e_max = 1e5
        i = 1
        while t < t_max:
            # keep time steps within dt_min, dt_max
            if dt < dt_min: 
                dt = dt_min 
            elif dt > dt_max: 
                dt = dt_max
            # compute rfk4, rfk5 and e = |rfk4 - rfk5|
            k1 = dt*f(t, u, strength, density)
            k2 = dt*f(t + (1/4)*dt, u + (1/4)*k1, strength, density)
            k3 = dt*f(t + (3/8)*dt, u + (3/32)*k1 + (9/32)*k2, strength, density)
            k4 = dt*f(t + (12/13)*dt, u + (1932)*k1 - (7200/2197)*k2 + (7296/2197)*k3, strength, density)
            k5 = dt*f(t + 1*dt, u + (439/216)*k1 -8*k2 + (3680/513)*k3 - (845/4104)*k4, strength, density)
            k6 = dt*f(t + (1/2)*dt, u + -(8/27)*k1 +2*k2 - (3544/2565)*k3 +(1859/4104)*k4 -(11/40)*k5, strength, density)
            u_5 = u + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 -(9/50)*k5 + (2/55)*k6
            u_4 = u + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 -(1/4)*k5
            e = np.linalg.norm(u)
            print(i) 
            i += 1
            print(e)
            # if error is too big, reject step and make dt smaller
            if (e > e_max and dt > dt_min): 
                dt /= 2
            # else accept the step
            else: 
                t += dt
                t_all.append(t)                 
                U_all.append(u_5) 
                if (e < e_min): # if error *too* small, make dt bigger
                    dt *= 2
            if u[3] < 0:
                break
            if u[1] < 0:
                break    
        return np.array(U_all), np.array(t_all)


    def forward_euler(self, f, initial_state, strength, density, t0=0.0, t_max=20000, dt=0.05):
        '''
            Return velocity, mass, angle, altitude, distance, radius (in order) calculation for each time step 
                    using Forward Euler
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
            u = u + dt*f(t, u, strength, density)  # euler guess
            U_all.append(u)
            t = t + dt
            t_all.append(t)
            if u[3] < 0:
                break
            if u[1] < 0:
                break   
        return np.array(U_all), np.array(t_all)
    
    def compare_analytical_numerical(self, radius, velocity, density, strength, angle, 
        init_altitude=100e3, n=100, radians=False):
        """
            Provides comparison analytical solutions and numerical to the ODEs in simplified scenario

            Parameters
            ---------
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

            init_altitude : float, optional
                Initial altitude in m

            n: integer, optional
                number of data points in analytical solution to compute 
                
            radians : logical, optional
                Whether angles should be given in degrees or radians. Default=False
                Angles returned in the DataFrame will have the same units as the
                input

            Returns
            -------
            five arrays :
            
            v  : velocity profile from analytical method
            
            z  : altitude profile from analytical method
            
            state_numerical  : velocity, mass, angle ,altitude, distance, radius, profile from numerical method
                                using gaspra solver
            
            time_numerical  : time profile from numerical method using gaspra solver
            
            result_sci : velocity, mass, angle ,altitude, distance, radius, profile time profile from numerical method using 
                        Scipy built RK-45 method
        """
        if radians == False: 
            angle *= np.pi/180        
        # calculating constants 
        A = np.pi*radius*radius
        # mass = density * volume
        m = density*A*(4/3)*radius
        beta = -self.H*self.Cd*A/(2*m*np.sin(angle)) 
        C = beta * np.exp(-init_altitude/self.H) - np.log(velocity)

        # Now computing v against height z
        z = np.linspace(init_altitude, 0, n)
        v = np.exp(beta*np.exp(-z/self.H) -C)
        
        ## Numerical part
        t_max = 20000
        tol = 1e-6
        initial_state = np.array([velocity, density*4/3*np.pi*radius**3, angle, init_altitude, 0, radius]) 
        state_numerical, time_numerical = self.RK_45_analytical(self.f_com_analytical, initial_state)
        result_sci = solve_ivp(self.f_com_analytical, (0, t_max), initial_state, t_eval=None, method='RK45', atol=tol, rtol=tol)
    
        ## Plotting part
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(z,v, 'ko', label = 'Analytical')
        ax.plot(state_numerical[:,3], state_numerical[:,0], 'bx', label = 'User created RK-45')
        ax.plot(result_sci.y[3], result_sci.y[0], 'r', label = 'Scipy Built-in RK-45 ')
        ax.set_xlim(state_numerical[-1,3], state_numerical[0,3])
        ax.set_xlabel('v')
        ax.set_ylim(state_numerical[-1,0], state_numerical[0,0])
        ax.set_ylabel('z')
        ax.legend(loc='best')
        plt.show()
        return z, v, state_numerical, time_numerical, result_sci     

    def f_com_analytical(self, t, state):
        '''
        Composite of derivative function of velocity, mass, angle, altitude, distance, radius (in order)
        Parameters in simple scenario
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
        f[0] = (-self.Cd * self.rhoa(z)* A * v**2) / (2 * m)
        f[1] = 0
        f[2] = 0
        f[3] = -v * np.sin(theta)
        f[4] = v * np.cos(theta)
        f[5] = 0
        return f

    def RK_45_analytical(self, f, initial_state,  t0=0.0, t_max=20000, dt=0.05 ):
            '''
            Return velocity, mass, angle, altitude, distance, radius (in order) calculation for each time step 
            using Runge_Kutta 4th order in simple scenario
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
                k1 = dt*f(t, u)
                k2 = dt*f(t + 0.5*dt, u + 0.5*k1)
                k3 = dt*f(t + 0.5*dt, u + 0.5*k2)
                k4 = dt*f(t + dt, u + k3)
                u = u + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
                U_all.append(u)
                t = t + dt
                t_all.append(t)
                if u[3] < 0:
                    break
                if u[1] < 0:
                    break    
            return np.array(U_all), np.array(t_all)


    def plot_analysis(self, result): 
        '''
        Return calculated parameters in data frame and make plot from it
        Parameters
        ----------
        result : data frame
            data frame containing velocity, mass, angle, altitude, distance, radius, time, and dedz (in order)
    
        Returns
        -------------
        Seven chart with different parameters based on data
        '''
        fig, axes = plt.subplots(3, 2, figsize=(12,8), sharex=True)
        result.plot(kind='line', x='altitude', y='velocity',ax = axes[0,0], title = 'velocity', grid=True)
        result.plot(kind='line', x='altitude', y='mass',ax = axes[0,1], title = 'mass', grid=True)
        result.plot(kind='line', x='altitude', y='angle',ax = axes[1,0], title = 'angle', grid=True)
        result.plot(kind='line', x='altitude', y='distance',ax = axes[1,1], title = 'distance', grid=True)
        result.plot(kind='line', x='altitude', y='radius',ax = axes[2,0], title = 'radius', grid=True)
        result.plot(kind='line', x='altitude', y='time',ax = axes[2,1], title = 'time', grid=True)
        plt.show()
        ax1 = plt.gca()
        max_row_id = result['dedz'].idxmax()
        max_dedz = result['dedz'].max()
        max_dedz_altitude = result['altitude'][max_row_id]
        result.plot(kind='line', x='altitude', y='dedz',ax = ax1, grid=True, figsize=(12,6))
        plt.plot(max_dedz_altitude, max_dedz, 'ro' )
        plt.annotate(('(%g , %.2f)' %(max_dedz_altitude, max_dedz)), xy=(max_dedz_altitude, max_dedz))
        plt.grid(True)
        plt.show() 


# earth = Planet(atmos_func='tabular')
# result, outcome = earth.impact(10, 20000, 3000, 10e5, 45)
# print(result)
# print(outcome)
# a = earth.plot_analysis(result)
