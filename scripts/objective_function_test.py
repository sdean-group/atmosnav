from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial 

"""
This script performs trajectory optimization on the original plan with
more complex objective function
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
wind_inst = WindFromData.from_data(DATA_PATH, start_time=START_TIME, integration_time_step=INTEGRATION_TIME_STEP)

# Create an agent

def make_weather_balloon(init_lat, init_lon, start_time, waypoint_time_step, integration_time_step, seed):
    return Airborne(
        jnp.array([ init_lat, init_lon, 0.0, 0.0 ]),
        PlanToWaypointController(start_time=start_time, waypoint_time_step=waypoint_time_step),
        SimpleAltitudeModel())

SEED = 0 
balloon = make_weather_balloon(
    42.4410187, -76.4910089, 
    START_TIME, WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP, 
    SEED)

# Create a plan
WAYPOINT_COUNT = 40 #  Total sim time = Waypoint Count * Waypoint Time Step = 40 * 3 hours = 5 days

def make_initial_plan(waypoint_count): 
    uppers = 10 + jnp.sin(2*np.pi*np.arange(waypoint_count)/10)
    lowers = uppers - 3
    return np.vstack([lowers,uppers]).T

def make_constant_plan(upper, lower, horizon_time):
    waypoint_count = horizon_time//WAYPOINT_TIME_STEP
    return np.vstack([ np.full((waypoint_count, ), lower), np.full((waypoint_count, ), upper)]).T

def test_plan(horizon_time):
    return make_constant_plan(1.0, 3.0, horizon_time)



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

    return jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, log))


@jax.jit
@partial(jax.grad, argnums=2)
def gradient_at(start_time, balloon, plan, wind):
    N = (len(plan)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    cost = 0.0
    def inner_run(i, time_balloon_cost):
        time, balloon, cost = time_balloon_cost
        # step the agent in time
        next_balloon, _ = balloon.step(time, plan, wind.get_direction(time, balloon.state))

        # cost += (next_balloon.state[0] - 40.416775)**2 + (next_balloon.state[1] + 3.703790)**2
        cost += (next_balloon.state[0] - 42.4440)**2 + (next_balloon.state[1] + 76.5019)**2
        # cost += (next_balloon.state[0] - 53.33306)**2 + (next_balloon.state[1] + 6.24889)**2
        # 53.33306 -6.24889
        # jump dt
        next_time = time + INTEGRATION_TIME_STEP

        return next_time, next_balloon, cost

    final_time, final_balloon, final_cost = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, cost))
    return final_cost

@jax.jit
def optimize_plan(start_time, balloon, plan, wind):
    def inner_opt(i, plan):
        d_plan = gradient_at(start_time, balloon, plan, wind)
        return plan - 0.5 * d_plan / jnp.linalg.norm(d_plan)
    return jax.lax.fori_loop(0, 1000, inner_opt, init_val=plan)

@partial(jax.jit, static_argnums=(1,)) #changing time_elapsed causes recompilation because array sizes must be known statically
def receeding_horizon_control(start_time, time_elapsed, balloon, observed_wind, true_wind):
    horizon_time = 60*60*24 # 1 day
    follow_time = 60*60*12 # 12 hours

    N = (time_elapsed//INTEGRATION_TIME_STEP)
    log = {
        't': jnp.zeros((N, ), dtype=jnp.int32),
        'h': jnp.zeros((N, )), 
        'lat': jnp.zeros((N, )), 
        'lon': jnp.zeros((N, )),
        'lbnd': jnp.zeros((N, )),
        'ubnd': jnp.zeros((N, ))}
    
    def inner_rhc(i, val):
        time, balloon, plan, log_idx, logs = val

        # Get the optimal plan given the 'observed' wind, but only follow part of it
        optimal_plan = optimize_plan(time, balloon, plan , observed_wind)
        plan_to_follow = optimal_plan[:follow_time//WAYPOINT_TIME_STEP]

        # run the part of the plan to follow in the real wind (the noisy wind)
        next_time, next_balloon, next_log = run(time, balloon, plan_to_follow, true_wind)
        next_balloon.controller.start_time += follow_time # let the balloon know it will run at a different time next iteration
        
        # store the unused part of the plan as an intial guess
        next_plan = jnp.concatenate((optimal_plan[follow_time//WAYPOINT_TIME_STEP:], test_plan(follow_time)))
        
        # add all the logged data into the big log dictionary
        n = len(next_log['t'])

        def inner_log_loop(i, loginfo):
            log_idx, logs = loginfo
            return log_idx+1, {
                't':logs['t'].at[log_idx].set(next_log['t'][i]),
                'h':logs['h'].at[log_idx].set(next_log['h'][i]),
                'lat':logs['lat'].at[log_idx].set(next_log['lat'][i]),
                'lon':logs['lon'].at[log_idx].set(next_log['lon'][i]),
                'lbnd':logs['lbnd'].at[log_idx].set(next_log['lbnd'][i]),
                'ubnd':logs['ubnd'].at[log_idx].set(next_log['ubnd'][i])}

        log_idx, next_log = jax.lax.fori_loop(0, n, inner_log_loop, (log_idx, logs))
        return next_time, next_balloon, next_plan, log_idx, next_log

    return jax.lax.fori_loop(0, time_elapsed//follow_time, inner_rhc, (start_time, balloon, test_plan(horizon_time), 0, log))[-1]


import time

initial_plan = make_initial_plan(WAYPOINT_COUNT)

if False:
    # Run initial plan
    _, _, log = run(START_TIME, balloon, initial_plan, wind_inst)
    tplt.plot_on_map(log)

    # Run optimized plan
    optimized_plan = optimize_plan(START_TIME, balloon, initial_plan, wind_inst)
    _, _, log = run(START_TIME, balloon, optimized_plan, wind_inst)
    tplt.plot_on_map(log)

# Run receeding horizon on plan
log = receeding_horizon_control(START_TIME, WAYPOINT_COUNT * WAYPOINT_TIME_STEP, balloon, wind_inst, wind_inst)
print(log)
tplt.plot_on_map(log)