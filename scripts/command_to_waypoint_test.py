from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np

"""
This script reads wind data and runs a weather balloon and runs simulations starting
from Ithaca while following a plan specified by up, down, stay commands

Runs many iterations of the same plan to show that it is fast.
"""

# The wind data directory
DATA_PATH = "../neotraj/data/proc/gfsanl/uv"

# initial time (as unix timestamp), must exist within data
START_TIME = 1691366400

# the numerical integration time step for the balloon 
INTEGRATION_TIME_STEP = 60*10

# The time between waypoint
COMMAND_TIME_STEP = 60*60*3

# Load wind data
wind_inst= WindFromData.from_data(DATA_PATH, start_time=START_TIME, integration_time_step=INTEGRATION_TIME_STEP)

# Create an agent

def make_weather_balloon(init_lat, init_lon, start_time, waypoint_time_step, integration_time_step, seed):
    return Airborne(
        jnp.array([ init_lat, init_lon, 0.0, 0.0 ]),
        CommandToWaypointController(),
        SimpleAltitudeModel() )
        # AltitudeModel(integration_time_step=integration_time_step, key=jax.random.key(seed)))

SEED = 0 
balloon = make_weather_balloon(
    42.4410187, -76.4910089, 
    START_TIME, COMMAND_TIME_STEP, INTEGRATION_TIME_STEP, 
    SEED)

# Create a plan
plan = jnp.array([ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2 ])

@jax.jit
def run(start_time, balloon, plan, wind):
    N = len(plan)
    log = {
        't': jnp.zeros((N, ), dtype=jnp.int32),
        'h': jnp.zeros((N, )), 
        'lat': jnp.zeros((N, )), 
        'lon': jnp.zeros((N, ))}

    def inner_run(i, time_and_balloon_and_log):
        time, balloon, log = time_and_balloon_and_log

        # step the agent in time
        next_balloon, info = balloon.step(time, plan[i], wind.get_direction(time, balloon.state))
        
        # update the log
        next_log = {
            't':log['t'].at[i].set(time.astype(int)),
            'h':log['h'].at[i].set(balloon.state[2]),
            'lat':log['lat'].at[i].set(balloon.state[0]),
            'lon':log['lon'].at[i].set(balloon.state[1])}
        
        # jump dt
        next_time = time + INTEGRATION_TIME_STEP

        return next_time, next_balloon, next_log

    return jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, log))[1:]


_, log = run(START_TIME, balloon, plan, wind_inst)
tplt.plot_lat_lon(log)
tplt.plot_altitude(log)