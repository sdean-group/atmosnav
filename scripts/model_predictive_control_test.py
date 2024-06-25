from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial 

"""
get_optimal_path
    do gradient ascent

while True:
    path = get_optimal_path()
    follow_path(path[:follow_time])

"""

# The wind data directory
WIND_DATA_PATH = "../neotraj/data/proc/gfsanl/uv"

# initial time (as unix timestamp), must exist within data
WIND_DATA_START_TIME = 1691366400

# the numerical integration time step for the balloon 
INTEGRATION_TIME_STEP = 60*10

# The time between waypoint
WAYPOINT_TIME_STEP = 60*60*3

# Load wind data
wind_inst = WindFromData.from_data(
    WIND_DATA_PATH, 
    start_time=WIND_DATA_START_TIME, 
    integration_time_step=INTEGRATION_TIME_STEP)

# Create an agent

def make_weather_balloon(init_lat, init_lon, start_time, waypoint_time_step, integration_time_step, seed):
    return Airborne(
        jnp.array([ init_lat, init_lon, 0.0, 0.0 ]),
        PlanToWaypointController(start_time=start_time, waypoint_time_step=waypoint_time_step),
        SimpleAltitudeModel())
        #AltitudeModel(integration_time_step=integration_time_step, key=jax.random.key(seed)))

SEED = 0 
balloon = make_weather_balloon(
    42.4410187, -76.4910089, 
    WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP, 
    SEED)

# Plan helper functions
def make_initial_plan(waypoint_count): 
    uppers = 10 + jnp.sin(2*np.pi*np.arange(waypoint_count)/10)
    lowers = uppers - 3
    return np.vstack([lowers,uppers]).T

def make_constant_plan(upper, lower, horizon_time):
    waypoint_count = horizon_time//WAYPOINT_TIME_STEP
    return np.vstack([ np.full((waypoint_count, ), lower), np.full((waypoint_count, ), upper)]).T

# Sim functions

@jax.jit
def trajectory_at(start_time, balloon, plan, wind):
    jax.debug.print("len(plan) is {x}", x=len(plan))
    N = (len(plan)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    jax.debug.print("N is {x}", x=N)
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

    return jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, log))


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
    return final_balloon.state[1] # final longitude

@jax.jit
def get_optimal_plan(start_time, balloon, plan, wind):
    # print(start_time)
    jax.debug.print("{x}", x=start_time)
    def inner_opt(i, stuff):
        time, balloon, plan = stuff
        # jax.debug.print('optimizing =D')
        d_plan = gradient_at(time, balloon, plan, wind)
        # jax.debug.print("{x}", x=d_plan)
        return time, balloon, plan + 0.5 * d_plan / jnp.linalg.norm(d_plan)
    return jax.lax.fori_loop(0, 200, inner_opt, init_val=(start_time, balloon, plan))[-1]



def test_plan(horizon_time):
    return make_constant_plan(1.0, 3.0, horizon_time)
    # return make_initial_plan(horizon_time//WAYPOINT_TIME_STEP)

def receeding_horizon_control(start_time, time_elapsed, balloon, wind):
    horizon_time = 60*60*24 # 1 day
    follow_time = 60*60*12 # 3 hours
    initial_plan = test_plan(horizon_time)

    logs=[]

    time = start_time
    
    while time < start_time + time_elapsed:
        optimal_plan = get_optimal_plan(time, balloon, initial_plan , wind)
        plan_to_follow = optimal_plan[:follow_time//WAYPOINT_TIME_STEP]
        # plan_to_follow = optimal_plan
        next_time, next_balloon, log = trajectory_at(balloon, plan_to_follow, wind)
        wind.start_time += horizon_time
        # last_plan = optimal_plan
        balloon = next_balloon
        time = next_time

        logs.append(log)

    return logs

ELAPSED_TIME = 60*60*24*3

print("without mpc")
tplt.plot_on_map(trajectory_at(START_TIME, balloon, get_optimal_plan(START_TIME, balloon, test_plan(ELAPSED_TIME), wind_inst), wind_inst)[-1])
# print(START_TIME)

print("running mpc")
logs = receeding_horizon_control(START_TIME, ELAPSED_TIME, balloon, wind_inst)
print("finished, showing plots")


concat_log = {
    't': [],
    'h': [], 
    'lat': [], 
    'lon': [],
    'lbnd':[],
    'ubnd': []}

for log in logs:
    concat_log['t'] += log['t'].tolist()
    concat_log['h'] += log['h'].tolist()
    concat_log['lat'] += log['lat'].tolist()
    concat_log['lon'] += log['lon'].tolist()
    concat_log['lbnd'] += log['lbnd'].tolist()
    concat_log['ubnd'] += log['ubnd'].tolist()

tplt.plot_on_map(concat_log)
        