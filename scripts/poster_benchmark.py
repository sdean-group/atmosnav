from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np
import time
import random

"""
This script reads wind data and runs a weather balloon and runs simulations starting
from Ithaca while following a plan.

Runs many iterations of the same plan to show that it is fast as running plans. Prints out
the compilation time and the run time for some number of iterations.
"""

# The wind data directory
DATA_PATH = "../data/small"

# initial time (as unix timestamp), must exist within data
START_TIME = 1691366400

# the numerical integration time step for the balloon 
INTEGRATION_TIME_STEP = 60*10

# The time between waypoint
WAYPOINT_TIME_STEP = 60*60*3

# Load wind data
start = time.time()
print('Loading wind data... ', end='')
wind_inst = WindFromData.from_data(DATA_PATH, INTEGRATION_TIME_STEP)
print(f'Took {time.time() - start}s')

# Create an agent

def make_weather_balloon(init_lat, init_lon, start_time, waypoint_time_step, integration_time_step, seed):
    return Airborne(
        jnp.array([ init_lat, init_lon, 0.0, 0.0 ]),
        PlanToWaypointController(start_time=start_time, waypoint_time_step=waypoint_time_step),
        AltitudeModel(integration_time_step=integration_time_step, key=jax.random.key(seed)))

SEED = 0 
balloon = make_weather_balloon(
    42.4410187, -76.4910089, 
    START_TIME, WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP, 
    SEED)

# Create a plan
WAYPOINT_COUNT = 40 #  Total sim time = Waypoint Count * Waypoint Time Step = 40 * 3 hours = 5 days
key = jax.random.key(seed=0)

def make_plan(waypoints, vertical_offset):
    uppers = vertical_offset + jnp.sin(2*np.pi*jnp.arange(waypoints)/10)
    lowers = uppers - 3
    return jnp.vstack([lowers,uppers]).T

# Create simulator

sim = DifferentiableSimulator(WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP)

start = time.time()
plan = make_plan(WAYPOINT_COUNT, 10)
print('Compiling code... ', end='')
_, log = sim.trajectory_at(START_TIME, balloon, plan, wind_inst)
print(f'Took {time.time() - start}s')

random.seed(0)

for j in range(10):
    elapsed = 0.0
    N = (j+1) * 1000
    print(f'Running {N} iterations... ', end='')
    logs = []
    for i in range(N):
        offset = random.uniform(3, 22)
        plan = make_plan(WAYPOINT_COUNT, offset)
        start = time.time()
        _, log = sim.trajectory_at(START_TIME, balloon, make_plan(WAYPOINT_COUNT, offset), wind_inst)
        logs.append(log)
        elapsed += (time.time() - start)

    print(f'Took {elapsed}s, {elapsed/N}s per iteration')
    # tplt.plot_on_map_many(logs)
    # tplt.plot_on_map_scan(logs[0], 'out.png')