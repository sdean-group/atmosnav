import jax.test_util
from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np

"""
This script performs trajectory optimization on the original plan.
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
wind_inst = WindFromData.from_data(DATA_PATH, integration_time_step=INTEGRATION_TIME_STEP)

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


import time

# Create a plan
WAYPOINT_COUNT = 40 #  Total sim time = Waypoint Count * Waypoint Time Step = 40 * 3 hours = 5 days
uppers = 10 + jnp.sin(2*np.pi*np.arange(WAYPOINT_COUNT)/10)
lowers = uppers - 3
plan = np.vstack([lowers,uppers]).T

# Create simulator
sim = DifferentiableSimulator(WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP)

@jax.jit
def optimize_plan(sim, start_time, balloon, plan, wind, steps, objective):
    def inner_opt(i, plan):
        d_plan = sim.gradient_at(start_time, balloon, plan, wind, objective)
        return plan + 0.5 * d_plan / jnp.linalg.norm(d_plan)
    return jax.lax.fori_loop(0, steps, inner_opt, init_val=plan)

class FinalLongitude(Objective):
    def evaluate(self, final_time, final_balloon):
        return final_balloon.state[1]

    def tree_flatten(self): 
        return tuple(), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return FinalLongitude()
    
obj = FinalLongitude()

#### 

JIT_LOOP = False

_, log = sim.trajectory_at(START_TIME, balloon, plan, wind_inst)
tplt.plot_on_map(log)


print('Compiling code... ', end='')
start = time.time()

if JIT_LOOP:
    ignore = optimize_plan(sim, START_TIME, balloon, plan, wind_inst, 1, obj)
else:
    ignore = sim.gradient_at(START_TIME, balloon, plan, wind_inst, obj)
print(f'Took {time.time() - start}s')

    

N = 1000
print(f'Running {N} steps of optimization...', end='')
start = time.time()
if JIT_LOOP:

    plan = optimize_plan(sim, START_TIME, balloon, plan, wind_inst, N, obj)
    
else:
        
    for i in range(N):
        d_plan = sim.gradient_at(START_TIME, balloon, plan, wind_inst, obj)
        plan = plan + 0.5 * d_plan / np.linalg.norm(d_plan)

total = time.time() - start
print(f'Took: {total}s, {total/N}s per iteration')

_, log = sim.trajectory_at(START_TIME, balloon, plan, wind_inst)
tplt.plot_on_map(log)