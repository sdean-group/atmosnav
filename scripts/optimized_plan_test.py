from atmosnav import *
from atmosnav.wind import Wind
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np

"""
This script performs trajectory optimization on the original plan.
"""

# The wind data directory
DATA_PATH = "../neotraj/data/proc/gfsanl/uv"

# initial time (as unix timestamp), must exist within data
START_TIME = 1691366400

# the numerical integration time step for the balloon 
INTEGRATION_TIME_STEP = 60*10

# The time between waypoint
WAYPOINT_TIME_STEP = 60*60*3

# Load wind data
wind = Wind.from_data(DATA_PATH, start_time=START_TIME, integration_time_step=INTEGRATION_TIME_STEP)

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
uppers = 10 + jnp.sin(2*np.pi*np.arange(WAYPOINT_COUNT)/10)
lowers = uppers - 3
plan = np.vstack([lowers,uppers]).T



@jax.jit
def run(start_time, balloon, plan, wind):
    N = len(plan)*WAYPOINT_TIME_STEP//INTEGRATION_TIME_STEP
    log = {
        't': jnp.zeros((N, ), dtype=jnp.int32),
        'h': jnp.zeros((N, )), 
        'lat': jnp.zeros((N, )), 
        'lon': jnp.zeros((N, )),
        'lbnd': jnp.zeros((N, )),
        'ubnd': jnp.zeros((N, ))}

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
            'lbnd':log['lbnd'].at[i].set(info['control_input'][0]),
            'ubnd':log['ubnd'].at[i].set(info['control_input'][1])}
        
        # jump dt
        next_time = time + INTEGRATION_TIME_STEP

        return next_time, next_balloon, next_log

    return jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, log))[1:]



from functools import partial 
@jax.jit
@partial(jax.grad, argnums=2)
def gradient_at(start_time, balloon, plan, wind):
    N = (len(plan)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    def inner_run(i, time_balloon):
        time, balloon = time_balloon
        # step the agent in time
        next_balloon, _ =balloon.step(time, plan, wind.get_direction(time, balloon.state))

        # jump dt
        next_time = time + INTEGRATION_TIME_STEP

        return next_time, next_balloon

    final_time, final_balloon = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon))
    return final_balloon.state[1]

# _, log = run(START_TIME, balloon, plan, wind)
# tplt.plot_on_map(log)

for i in range(100):
    print(i)
    d_plan = gradient_at(START_TIME, balloon, plan, wind)
    plan = plan + 0.5 * d_plan / np.linalg.norm(d_plan)


# _, log = run(START_TIME, balloon, plan, wind)
# tplt.plot_on_map(log)