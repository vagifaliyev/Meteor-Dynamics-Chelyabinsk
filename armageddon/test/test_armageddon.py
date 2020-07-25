
from collections import OrderedDict
import pandas as pd

from pytest import fixture

# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly

@fixture(scope='module')
def armageddon():
    """Perform the module import"""
    import armageddon
    return armageddon

@fixture(scope='module')
def planet(armageddon):
    """Return a default planet with a constant atmosphere"""
    return armageddon.Planet(atmos_func='constant')

@fixture(scope='module')
def input_data():
    input_data = {'radius': 1.,
                  'velocity': 1.0e5,
                  'density': 3000.,
                  'strength': 1e32,
                  'angle': 30.0,
                  'init_altitude':100e3,
                  'dt': 0.05,
                  'radians': False 
                 }
    return input_data

@fixture(scope='module')
def result(planet, input_data):
    """Solve a default impact for the default planet"""

    result = planet.solve_atmospheric_entry(**input_data)

    return result

def test_import(armageddon):
    """Check package imports"""
    assert armageddon

# def test_planet_signature(armageddon):
#     """Check planet accepts specified inputs"""
#     inputs = OrderedDict(atmos_func='constant',
#                          atmos_filename=None,
#                          Cd=1., Ch=0.1, Q=1e7, Cl=1e-3,
#                          alpha=0.3, Rp=6371e3,
#                          g=9.81, H=8000., rho0=1.2)

#     # call by keyword
#     planet = armageddon.Planet(**inputs)

#     # call by position
#     planet = armageddon.Planet(*inputs.values())

def test_attributes(planet):
    """Check planet has specified attributes."""
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'alpha', 'Rp', 'g', 'H', 'rho0'):
        assert hasattr(planet, key)

def test_solve_atmospheric_entry(result, input_data):
    """Check atmospheric entry solve. 

    Currently only the output type for zero timesteps."""
    
    assert type(result) is pd.DataFrame
    
    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns

    assert result.velocity.iloc[0] == input_data['velocity']
    assert result.angle.iloc[0] == input_data['angle']
    assert result.altitude.iloc[0] == input_data['init_altitude']
    assert result.distance.iloc[0] == 0.0
    assert result.radius.iloc[0] == input_data['radius']
    assert result.time.iloc[0] == 0.0


def test_calculate_energy(planet, result):

    energy = planet.calculate_energy(result=result)

    print(energy)

    assert type(energy) is pd.DataFrame
    
    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time', 'dedz'):
        assert key in energy.columns

def test_analyse_outcome(planet, result):

    result = planet.calculate_energy(result.copy())
    outcome = planet.analyse_outcome(result)
    
    assert type(outcome) is dict    

def test_ensemble(planet, armageddon):

    fiducial_impact = {'radius': 0.0,
                       'angle': 0.0,
                       'strength': 0.0,
                       'velocity': 0.0,
                       'density': 0.0}
    
    ensemble = armageddon.ensemble.solve_ensemble(planet,
                                                  fiducial_impact,
                                                  variables=[], radians=False,
                                                  rmin=8, rmax=12)

    assert 'burst_altitude' in ensemble.columns

# def test_marsdensity(z):
#     # cheking mars denisty values against hand calculations
#     assert marsdensity(8000) = approx(0.00763628504)
#     assert marsdensity(7000) = approx(0.00827621144)
#     assert marsdensity(5000) = approx(0.00978514855)

