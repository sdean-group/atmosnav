from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np

"""
This script performs trajectory optimization on the original plan.
"""

# The wind data directory
DATA_PATH = "/Users/bradleyguo/Python Projects/atmosnav/atmosnav/data/proc/gfsanl/uv"

# initial time (as unix timestamp), must exist within data
START_TIME = 1691366400

# the numerical integration time step for the balloon 
INTEGRATION_TIME_STEP = 60*10

# The time between waypoint
WAYPOINT_TIME_STEP = 60*60*3

# Start and end
start = (42.4410187, -76.4910089, 12)
#destination = (42.4410187, -76.4910089, 0) #Ithaca
#destination = (40.4168, -3.7038, 0) #Madrid
#destination = (64.1470, -21.9408, 0) #Reykjavik
destination = (44.3876, -68.2039, 0) #Bar Harbor


# Create a plan
WAYPOINT_COUNT = 40 #  Total sim time = Waypoint Count * Waypoint Time Step = 40 * 3 hours = 5 days
# uppers = 10 + jnp.sin(2*np.pi*np.arange(WAYPOINT_COUNT)/10)
# lowers = uppers - 3
# plan = np.vstack([lowers,uppers]).T
plan = 10*np.ones((WAYPOINT_COUNT, 1))

# Load wind data
wind = WindFromData.from_data(DATA_PATH, start_time=START_TIME, integration_time_step=INTEGRATION_TIME_STEP)


# Create an agent

def make_weather_balloon(init_lat, init_lon, init_height, start_time, waypoint_time_step, integration_time_step, seed):
    return Airborne(
        jnp.array([ init_lat, init_lon, init_height, 0.0 ]),
        PlanToWaypointController(start_time=start_time, waypoint_time_step=waypoint_time_step),
        DeterministicAltitudeModel(integration_time_step=integration_time_step))
        #AltitudeModel(integration_time_step=integration_time_step, key=jax.random.key(seed)))

SEED = 0 
balloon = make_weather_balloon(
    start[0], start[1], start[2], 
    START_TIME, WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP, 
    SEED)


@jax.jit
def run(start_time, balloon, plan, wind):
    N = len(plan)*WAYPOINT_TIME_STEP//INTEGRATION_TIME_STEP
    log = {
        't': jnp.zeros((N, ), dtype=jnp.int32),
        'h': jnp.zeros((N, )), 
        'lat': jnp.zeros((N, )), 
        'lon': jnp.zeros((N, )),
        # 'lbnd': jnp.zeros((N, )),
        # 'ubnd': jnp.zeros((N, ))}
        'plan': jnp.zeros((N, ))}

    def inner_run(i, time_and_balloon_and_log):
        time, balloon, log = time_and_balloon_and_log

        # step the agent in time
        next_balloon, info = balloon.step(time, plan, wind.get_direction(time, balloon.state))
        
        # update the log
        next_log = {
            't':log['t'].at[i].set(time.astype(int)),
            'h':log['h'].at[i].set(balloon.state[2]),
            'lat':log['lat'].at[i].set(balloon.state[0]),
            'lon':log['lon'].at[i].set(balloon.state[1]),
            # 'lbnd':log['lbnd'].at[i].set(info['control_input'][0]),
            # 'ubnd':log['ubnd'].at[i].set(info['control_input'][1])}
            'plan':log['plan'].at[i].set(info['control_input'][0])}
        
        # jump dt
        next_time = time + INTEGRATION_TIME_STEP

        return next_time, next_balloon, next_log

    return jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, log))[1:]



discount = 0.99
from functools import partial 
@jax.jit
@partial(jax.grad, argnums=2)
def gradient_at(start_time, balloon, plan, wind):
    N = (len(plan)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    def inner_run(i, time_balloon_cost):
        time, balloon, cost = time_balloon_cost
        cost += -discount**(N-1-i)*((balloon.state[0]-destination[0])**2 + (balloon.state[1]-destination[1])**2 + (balloon.state[2]-destination[2])**2) 
        # step the agent in time
        next_balloon, _ =balloon.step(time, plan, wind.get_direction(time, balloon.state))

        # jump dt
        next_time = time + INTEGRATION_TIME_STEP    

        return next_time, next_balloon, cost

    final_time, final_balloon, cost = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, 0))
    #return -final_balloon.state[1]
    #return -((final_balloon.state[0]-destination[0])**2 + (final_balloon.state[1]-destination[1])**2 + (final_balloon.state[2])**2) 
    return cost

import time

JIT_LOOP = True

_, log = run(START_TIME, balloon, plan, wind)
# tplt.plot_on_map(log)
tplt.deterministic_plot_on_map(log)

start = time.time()
if JIT_LOOP:

    @jax.jit
    def optimize_plan(start_time, balloon, plan, wind):
        def inner_opt(i, plan):
            d_plan = gradient_at(start_time, balloon, plan, wind)
            return plan + 0.5 * d_plan / jnp.linalg.norm(d_plan)
        return jax.lax.fori_loop(0, 10000, inner_opt, init_val=plan)

    plan = optimize_plan(START_TIME, balloon, plan, wind)
    
else:
        
    for i in range(1000):
        d_plan = gradient_at(START_TIME, balloon, plan, wind)
        plan = plan + 0.5 * d_plan / np.linalg.norm(d_plan)



print(f'Took: {time.time() - start} s')

_, log = run(START_TIME, balloon, plan, wind)
# tplt.plot_on_map(log)
tplt.deterministic_plot_on_map(log)