import numpy as np
import pandas as pd

class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='constant', atmos_filename=None,
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
            self.rhoa = lambda x: rho0 * np.exp(-z / H)
        elif atmos_func == 'tabular':
            raise NotImplementedError
        elif atmos_func == 'mars':
            raise NotImplementedError
        elif atmos_func == 'constant':
            self.rhoa = lambda x: rho0
        else:
            raise NotImplementedError

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

        # Enter your code here
        raise NotImplementedError

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

        # Enter your code here to solve the differential equations

        return pd.DataFrame({'velocity': velocity,
                             'mass': np.nan,
                             'angle': angle,
                             'altitude': init_altitude,
                             'distance': 0.0,
                             'radius': radius,
                             'time': 0.0}, index=range(1))

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

        # Replace these lines with your code to add the dedz column to
        # the result DataFrame
        result = result.copy()
        result.insert(len(result.columns),
                      'dedz', np.array(np.nan))

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
        outcome = {}
        return outcome

    def f_com(self, t, state):
        f = np.zeros_like(state)
        v, m, theta, z, x, r = state
        A = np.pi * r**2
        # derivative function in order : [0] dv [1] dm [2] dtheta [3] dz [4] dx [5] dr
        f[0] = ((-self.Cd * self.rhoa(x) * A * v**2) / (2 * m)) + (self.g * np.sin(theta))
        f[1] = (-self.Ch * self.rhoa(x) * A * v**3) / (2* self.Q)
        f[2] = (self.g * np.cos(theta) / v) - (self.Cl * self.rhoa(x) * A * v / (2 * m)) - (v * np.cos(theta) / (self.Rp + z))
        f[3] = -v * np.sin(theta)
        f[4] = (v * np.cos(theta)) / (1 + (z / self.Rp))
        f[5] = np.sqrt((7/2 * self.alpha * self.rhoa(x) / density)) * v
        return f


    def RK_45(self, f, initial_state, t0=0.0, t_max=10, dt=0.05):
        u = np.array(initial_state)
        U_all = np.array(u)
        t = np.array(t0)
        t_all = np.array(t0) 
        z = u[3]                
        while z > 0:
            state = u
            k1 = dt*f(t, u)
            k2 = dt*f(t + 0.5*dt, u + 0.5*k1)
            k3 = dt*f(t + 0.5*dt, u + 0.5*k2)
            k4 = dt*f(t + dt, u + k3)
            u = u + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
            U_all = np.vstack((U_all,u))
            t = t + dt
            t_all = np.vstack((t_all,t))
            z = u[3]
        return U_all, t_all

    def forward_euler(self, f, initial_state, t0=0.0, t_max=10, dt=0.01):
        """ Forward Euler time-stepper.
        f = f(t,y) is the RHS function.
        u0 is the initial condition.
        t0 is the initial time; t_max is the end time.
        dt is the time step size.
        """ 
        u = np.array(initial_state)
        U_all = np.array(u)
        t = np.array(t0)
        t_all = np.array(t0) 
        while t < t_max:
            state = u
            u = u + dt*f(t, u)  # euler guess
            U_all = np.vstack((U_all,u))
            t = t + dt
            t_all = np.vstack((t_all,t))
        return U_all, t_all

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

earth = Planet()
# initial state in order : [0] v [1] m [2] theta [3] z [4] x [5] r
initial_state = np.array([20000, 1.256637e7, earth.deg_to_rad(45), 100e3, 0, 10])
t0 = 0
t_max = 20
dt = 0.05
density = 3000

state, time = earth.RK_45(earth.f_com, initial_state, t0, t_max, dt)

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(6, 20))
fig.tight_layout(w_pad=5)  # add some padding otherwise the axes labels overlap

ax1.plot(time, state[:,0], 'b', label='v')
ax1.set_ylabel('$v$', fontsize=16)
ax2.plot(time, state[:,1], 'b', label='m')
ax2.set_ylabel('$m$', fontsize=16)
ax3.plot(time, state[:,2], 'b', label='theta')
ax3.set_ylabel('$theta$', fontsize=16)
ax4.plot(time, state[:,3], 'b', label='z')
ax4.set_ylabel('$z$', fontsize=16)
ax5.plot(time, state[:,4], 'b', label='x')
ax5.set_ylabel('$x$', fontsize=16)
ax6.plot(time, state[:,5], 'b', label='r')
ax6.set_ylabel('$r$', fontsize=16)

state_FE, time_FE = earth.forward_euler(earth.f_com, initial_state)

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(6, 20))
fig.tight_layout(w_pad=5)  # add some padding otherwise the axes labels overlap

ax1.plot(time_FE, state_FE[:,0], 'b', label='v')
ax1.set_ylabel('$v$', fontsize=16)
ax2.plot(time_FE, state_FE[:,1], 'b', label='m')
ax2.set_ylabel('$m$', fontsize=16)
ax3.plot(time_FE, state_FE[:,2], 'b', label='theta')
ax3.set_ylabel('$theta$', fontsize=16)
ax4.plot(time_FE, state_FE[:,3], 'b', label='z')
ax4.set_ylabel('$z$', fontsize=16)
ax5.plot(time_FE, state_FE[:,4], 'b', label='x')
ax5.set_ylabel('$x$', fontsize=16)
ax6.plot(time_FE, state_FE[:,5], 'b', label='r')
ax6.set_ylabel('$r$', fontsize=16)