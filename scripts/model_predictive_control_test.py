from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial 

"""

Runs gradient ascent trajectory optimization in a given wind field then runs it in a different wind field that
has added disturbances / noise.

Also does receeding horizon control inside the same observed and truth wind fields to see performance differences.


RCH: 
get_optimal_path
    do gradient ascent

while True:
    path = get_optimal_path()
    follow_path(path[:follow_time])

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
class WithDisturbance(Wind):
    def __init__(self, wind):
        self.wind = wind

    def get_direction(self, time: jnp.float32, state: Array) -> tuple[jnp.float32, jnp.float32]:
        dv, du = self.wind.get_direction(time, state) 
        dv_noise, du_noise = self.get_disturbance()
        return (dv+dv_noise, du+du_noise)
    
    def get_disturbance(self):
        return (0.05, 0)
    
    def tree_flatten(self):
        return (self.wind, ), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return WithDisturbance(*children)



wind_inst = WindFromData.from_data(DATA_PATH, INTEGRATION_TIME_STEP)

# Create an agent
def make_weather_balloon(init_lat, init_lon, start_time, waypoint_time_step, integration_time_step, seed):
    return Airborne(
        jnp.array([ init_lat, init_lon, 0.0, 0.0 ]),
        PlanToWaypointController(start_time=start_time, waypoint_time_step=waypoint_time_step),
        SimpleAltitudeModel())
        # AltitudeModel(integration_time_step=integration_time_step, key=jax.random.key(seed)))

SEED = 0 
balloon = make_weather_balloon(
    42.4410187, -76.4910089, 
    START_TIME, WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP, 
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
    N = (len(plan)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
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
    #return final_balloon.state[1] # final longitude
    return -((final_balloon.state[0]-42.4410187)**2 + (final_balloon.state[1]+76.4910089)**2)

@jax.jit
def get_optimal_plan(start_time, balloon, plan, wind):
    def inner_opt(i, stuff):
        time, balloon, plan = stuff
        d_plan = gradient_at(time, balloon, plan, wind)
        return time, balloon, plan + 0.5 * d_plan / jnp.linalg.norm(d_plan)
    return jax.lax.fori_loop(0, 1000, inner_opt, init_val=(start_time, balloon, plan))[-1]

def test_plan(horizon_time):
    return make_constant_plan(1.0, 3.0, horizon_time)

@partial(jax.jit, static_argnums=(1,)) #changing time_elapsed causes recompilation because array sizes must be known statically
def receeding_horizon_control(start_time, time_elapsed, balloon, observed_wind, true_wind):
    horizon_time = 60*60*24 # 1 day
    follow_time = 60*60*6 # 9 hours

    N = (time_elapsed//INTEGRATION_TIME_STEP)
    log = {
        't': jnp.zeros((N, ), dtype=jnp.int32),
        'h': jnp.zeros((N, )), 
        'lat': jnp.zeros((N, )), 
        'lon': jnp.zeros((N, )),
        'lbnd': jnp.zeros((N, )),
        'ubnd': jnp.zeros((N, ))}
    
    def inner_rhc(_, val):
        time, balloon, plan, log_idx, logs = val

        # Get the optimal plan given the 'observed' wind, but only follow part of it
        optimal_plan = get_optimal_plan(time, balloon, plan , observed_wind)
        plan_to_follow = optimal_plan[:follow_time//WAYPOINT_TIME_STEP]

        # run the part of the plan to follow in the real wind (the noisy wind)
        next_time, next_balloon, next_log = trajectory_at(time, balloon, plan_to_follow, true_wind)
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

# This exists here for reference
def unjitted_receeding_horizon_control(start_time, time_elapsed, balloon, observed_wind, true_wind):
    horizon_time = 60*60*24 # 1 day
    follow_time = 60*60*3 # 9 hours
    last_plan = test_plan(horizon_time)

    logs=[]

    time = start_time
    
    while time < start_time + time_elapsed:
        optimal_plan = get_optimal_plan(time, balloon, last_plan, observed_wind)
        plan_to_follow = optimal_plan[:follow_time//WAYPOINT_TIME_STEP]
        next_time, next_balloon, log = trajectory_at(time, balloon, plan_to_follow, true_wind)
        next_balloon.controller.start_time += follow_time
        
        logs.append(log)

        last_plan = np.concatenate((optimal_plan[follow_time//WAYPOINT_TIME_STEP:], test_plan(follow_time)))
        balloon = next_balloon
        time = next_time

    return logs

ELAPSED_TIME = 60*60*24*5


# Get the optimal plan in the observed wind data, but then run it in the real wind field
print("Without MPC...")
optimal_plan_no_noise = get_optimal_plan(START_TIME, balloon, test_plan(ELAPSED_TIME), wind_inst) 
_, _, log = trajectory_at(START_TIME, balloon, optimal_plan_no_noise, wind_inst)
print(log['lon'][-1])
tplt.plot_on_map(log)

# Runs receeding horizon control
print("Running MPC...")

USING_JITTED = True
if USING_JITTED:
    log = receeding_horizon_control(START_TIME, ELAPSED_TIME, balloon, wind_inst, wind_inst)
    print(log['lon'][-1])
    tplt.plot_on_map(log)
else:
    logs = unjitted_receeding_horizon_control(START_TIME, ELAPSED_TIME, balloon, wind_inst, wind_inst)
    tplt.plot_on_map_many(logs)