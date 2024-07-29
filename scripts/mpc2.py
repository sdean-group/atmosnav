from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial 
import optax
import matplotlib.pyplot as plt
import time
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
DATA_PATH = "/Users/bradleyguo/Python Projects/atmosnav/atmosnav/data/proc/gfsanl/uv"

# initial time (as unix timestamp), must exist within data
START_TIME = 1691366400

# the numerical integration time step for the balloon 
INTEGRATION_TIME_STEP = 60*10

# The time between waypoint
WAYPOINT_TIME_STEP = 60*60*3

wind_inst = WindFromData.from_data(DATA_PATH, start_time=START_TIME, integration_time_step=INTEGRATION_TIME_STEP)

# Load wind data
class WithDisturbance(Wind):
    def __init__(self, wind):
        self.wind = wind

    def get_direction(self, time: jnp.float32, state: Array) -> tuple[jnp.float32, jnp.float32]:
        dv, du = self.wind.get_direction(time, state) 
        a = state[0]
        b = state[1]
        h = jnp.clip(state[2], 0, 22)
        wind_factor = jnp.absolute((2.0/(1.0+jnp.exp(-20000.0*h)) - 1.0))
        dv_noise, du_noise = self.get_disturbance(jax.random.key(((a + b) * (a + b + 1) / 2 + a).astype(int)), wind_factor)
        return (dv+dv_noise, du+du_noise)

    def get_disturbance(self, key, wind_factor):
        key, subkey = jax.random.split(key)

        return 0.01*wind_factor*jax.random.normal(subkey, (2,))
    
    def tree_flatten(self):
        return (self.wind, ), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return WithDisturbance(*children)
    
# Create an agent
def make_weather_balloon(init_lat, init_lon, init_height, start_time, waypoint_time_step, integration_time_step, seed):
    return Airborne(
        jnp.array([ init_lat, init_lon, init_height, 0.0 ]),
        PlanToWaypointController(start_time=start_time, waypoint_time_step=waypoint_time_step),
        DeterministicAltitudeModel(integration_time_step=integration_time_step))
        #SimpleAltitudeModel())
        # AltitudeModel(integration_time_step=integration_time_step, key=jax.random.key(seed)))

#Start and end
start = (42.4410187, -76.4910089, 0)
#destination = (42.4410187, -76.4910089, 0) #Ithaca
#destination = (40.4168, -3.7038, 0) #Madrid
#destination = (64.1470, -21.9408, 0) #Reykjavik
#destination = (44.3876, -68.2039, 0) #Bar Harbor
#destination = (41.8781, -87.6298, 0) #Chicago
#destination = (51.5072, -0.1276, 0) #London
destination = (42.3601, -71.0589, 0) #Boston
#destination = (37.4419, -122.1430, 0) #Palo Alto


SEED = 0 
balloon = make_weather_balloon(
    start[0], start[1], start[2],
    START_TIME, WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP, 
    SEED)

# Plan helper functions
@partial(jax.jit, static_argnums=(3,))
def make_plan(start_time, balloon, wind, horizon_time, discount=0.99, vertical_weight=111):
    num_plans = 5000
    key = jax.random.key(time.time())
    WAYPOINT_COUNT = horizon_time//WAYPOINT_TIME_STEP+1
    init_plan = jnp.zeros((WAYPOINT_COUNT, 1))
    init_cost = cost_at(START_TIME, balloon, init_plan, wind)

    def add_plan(i, key_bestplancost):
        key, best_plan, best_cost = key_bestplancost
        key, *subkeys = jax.random.split(key, 3)

        plan = 22*jax.random.uniform(subkeys[0]) + jnp.sin(2*jnp.pi*jax.random.uniform(subkeys[1])*jnp.arange(WAYPOINT_COUNT)/10)
        plan = jnp.reshape(plan, (WAYPOINT_COUNT, 1))
        cost = cost_at(start_time, balloon, plan, wind, discount, vertical_weight)

        best_plan = jax.lax.cond(cost < best_cost,
                                 lambda op: op[0],
                                 lambda op: op[1],
                                 operand=(plan, best_plan))
        
        best_cost = jax.lax.cond(cost < best_cost,
                                 lambda op: op[0],
                                 lambda op: op[1],
                                 operand=(cost, best_cost))

        return (key, best_plan, best_cost)
    return jax.lax.fori_loop(0, num_plans, add_plan, init_val=(key, init_plan, init_cost))[1]

# Sim functions

@jax.jit
def trajectory_at(start_time, balloon, plan, wind):
    N = ((len(plan)-1)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    log = {
        't': jnp.zeros((N, ), dtype=jnp.int32),
        'h': jnp.zeros((N, )), 
        'lat': jnp.zeros((N, )), 
        'lon': jnp.zeros((N, )),
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
            'plan':log['plan'].at[i].set(info['control_input'][0])}
        
        # jump dt
        next_time = time + INTEGRATION_TIME_STEP

        return next_time, next_balloon, next_log

    return jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, log))

@jax.jit
def cost_at(start_time, balloon, plan, wind, discount=0.99, vertical_weight=111):
    N = (len(plan)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    def inner_run(i, time_balloon_cost):
        time, balloon, cost = time_balloon_cost
        cost += discount**(N-1-i)*((balloon.state[0]-destination[0])**2 + (balloon.state[1]-destination[1])**2 + (vertical_weight*(balloon.state[2]-destination[2])/111)**2) 
        # step the agent in time
        next_balloon, _ =balloon.step(time, plan, wind.get_direction(time, balloon.state))

        # jump dt
        next_time = time + INTEGRATION_TIME_STEP    

        return next_time, next_balloon, cost

    final_time, final_balloon, cost = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, 0))
    #return -final_balloon.state[1]
    #return ((final_balloon.state[0]-destination[0])**2 + (final_balloon.state[1]-destination[1])**2 + (final_balloon.state[2])**2) 
    return cost

from functools import partial 
@jax.jit
@partial(jax.grad, argnums=2)
def gradient_at(start_time, balloon, plan, wind, discount=0.99, vertical_weight=111):
    N = (len(plan)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    def inner_run(i, time_balloon_cost):
        time, balloon, cost = time_balloon_cost
        cost += discount**(N-1-i)*((balloon.state[0]-destination[0])**2 + (balloon.state[1]-destination[1])**2 + (vertical_weight*(balloon.state[2]-destination[2])/111)**2) 
        # step the agent in time
        next_balloon, _ =balloon.step(time, plan, wind.get_direction(time, balloon.state))

        # jump dt
        next_time = time + INTEGRATION_TIME_STEP    

        return next_time, next_balloon, cost

    final_time, final_balloon, cost = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, 0))
    #return -final_balloon.state[1]
    #return ((final_balloon.state[0]-destination[0])**2 + (final_balloon.state[1]-destination[1])**2 + (final_balloon.state[2])**2) 
    return cost

@jax.jit
def get_optimal_plan(start_time, balloon, plan, wind, discount=0.99, vertical_weight=111):

    optimizer = optax.adam(learning_rate = 0.03)
    opt_state = optimizer.init(plan)
    grad = gradient_at(start_time, balloon, plan, wind, discount, vertical_weight)

    def step(i, grad_opt_state_plan):
        grad, opt_state, plan = grad_opt_state_plan
        updates, opt_state = optimizer.update(grad, opt_state, plan)
        plan = optax.apply_updates(plan, updates)
        grad = gradient_at(start_time, balloon, plan, wind, discount, vertical_weight)

        return grad, opt_state, plan

    return jax.lax.fori_loop(0, 10000, step, init_val = (grad, opt_state, plan))[-1]

@partial(jax.jit, static_argnums=(1,)) #changing time_elapsed causes recompilation because array sizes must be known statically
def receeding_horizon_control(start_time, time_elapsed, balloon, observed_wind, true_wind):
    horizon_time = 60*60*24 # 1 day
    follow_time = 60*60*9 # 9 hours

    N = (time_elapsed//INTEGRATION_TIME_STEP)
    log = {
        't': jnp.zeros((N, ), dtype=jnp.int32),
        'h': jnp.zeros((N, )), 
        'lat': jnp.zeros((N, )), 
        'lon': jnp.zeros((N, )),
        'plan': jnp.zeros((N, ))}
    
    def inner_rhc(i, val):
        time, balloon, plan, log_idx, logs = val

        # Get the optimal plan given the 'observed' wind, but only follow part of it
        optimal_plan = get_optimal_plan(time, balloon, plan, observed_wind)
        plan_to_follow = optimal_plan[:follow_time//WAYPOINT_TIME_STEP+1]

        # run the part of the plan to follow in the real wind (the noisy wind)
        next_time, next_balloon, next_log = trajectory_at(time, balloon, plan_to_follow, true_wind)
        next_balloon.controller.start_time += follow_time # let the balloon know it will run at a different time next iteration
        
        # next guess
        next_plan = make_plan(next_time, next_balloon, observed_wind, horizon_time)
        
        # add all the logged data into the big log dictionary
        n = len(next_log['t'])

        def inner_log_loop(i, loginfo):
            log_idx, logs = loginfo
            return log_idx+1, {
                't':logs['t'].at[log_idx].set(next_log['t'][i]),
                'h':logs['h'].at[log_idx].set(next_log['h'][i]),
                'lat':logs['lat'].at[log_idx].set(next_log['lat'][i]),
                'lon':logs['lon'].at[log_idx].set(next_log['lon'][i]),
                'plan':logs['plan'].at[log_idx].set(next_log['plan'][i]),}

        log_idx, next_log = jax.lax.fori_loop(0, n, inner_log_loop, (log_idx, logs))

        return next_time, next_balloon, next_plan, log_idx, next_log
    
    return jax.lax.fori_loop(0, time_elapsed//follow_time, inner_rhc, (start_time, balloon, make_plan(start_time, balloon, observed_wind, horizon_time), 0, log))[-1]
    # follow_time = time_elapsed % follow_time
    # print(follow_time == 0)
    # return jax.lax.cond(follow_time == 0,
    #                     lambda op: op[-1],
    #                     lambda op: inner_rhc(0, op)[-1],
    #                     operand = val)

def unjitted_receeding_horizon_control(start_time, time_elapsed, balloon, observed_wind, true_wind):
    
    horizon_time = 60*60*24*1 # 1 day
    follow_time = 60*60*9 # 9 hours

    N = (time_elapsed//INTEGRATION_TIME_STEP)
    log = {
        't': jnp.zeros((N, ), dtype=jnp.int32),
        'h': jnp.zeros((N, )), 
        'lat': jnp.zeros((N, )), 
        'lon': jnp.zeros((N, )),
        'plan': jnp.zeros((N, ))}
    
    log_idx = 0
    time = start_time
    next_plan = make_plan(start_time, balloon, observed_wind, horizon_time)

    #Long Distance
    while time < start_time + time_elapsed and (balloon.state[0]-destination[0])**2 + (balloon.state[1]-destination[1])**2 + ((balloon.state[2]-destination[2]))**2 > 3:
        #discount = 0.01**(1./(horizon_time // INTEGRATION_TIME_STEP))
        optimal_plan = get_optimal_plan(time, balloon, next_plan, observed_wind)
        plan_to_follow = optimal_plan[:follow_time//WAYPOINT_TIME_STEP+1]
        
        next_time, next_balloon, next_log = trajectory_at(time, balloon, plan_to_follow, true_wind)
        next_balloon.controller.start_time = next_time
        
        next_plan = make_plan(next_time, next_balloon, observed_wind, horizon_time)
        

        balloon = next_balloon
        print((balloon.state[0]-destination[0])**2 + (balloon.state[1]-destination[1])**2 + ((balloon.state[2]-destination[2])/111)**2)

        time = next_time

        n = len(next_log['t'])

        for i in range(n):
            log = {
                't':log['t'].at[log_idx].set(next_log['t'][i]),
                'h':log['h'].at[log_idx].set(next_log['h'][i]),
                'lat':log['lat'].at[log_idx].set(next_log['lat'][i]),
                'lon':log['lon'].at[log_idx].set(next_log['lon'][i]),
                'plan':log['plan'].at[log_idx].set(next_log['plan'][i]),}
            log_idx+=1 


    #Short distance
    horizon_time = 60*60*6 # 3 hours
    follow_time = 60*60*3 # 3 hours

    next_plan = make_plan(time, balloon, observed_wind, horizon_time, discount=0.9)
    optimal_plan = get_optimal_plan(time, balloon, next_plan, observed_wind, discount=0.9)
    plan_to_follow = optimal_plan[:follow_time//WAYPOINT_TIME_STEP+1]



    while time < start_time + time_elapsed and (balloon.state[2] > 0 or jnp.any(plan_to_follow > 0)):
        #discount = 0.01**(1./(horizon_time // INTEGRATION_TIME_STEP))

        optimal_plan = get_optimal_plan(time, balloon, next_plan, observed_wind, discount=0.9)
        plan_to_follow = optimal_plan[:follow_time//WAYPOINT_TIME_STEP+1]

        next_time, next_balloon, next_log = trajectory_at(time, balloon, plan_to_follow, true_wind)
        next_balloon.controller.start_time = next_time
        
        next_plan = make_plan(next_time, next_balloon, observed_wind, horizon_time, discount=0.9)
        
        balloon = next_balloon
        print((balloon.state[0]-destination[0])**2 + (balloon.state[1]-destination[1])**2 + ((balloon.state[2]-destination[2])/111)**2)

        time = next_time

        n = len(next_log['t'])

        for i in range(n):
            log = {
                't':log['t'].at[log_idx].set(next_log['t'][i]),
                'h':log['h'].at[log_idx].set(next_log['h'][i]),
                'lat':log['lat'].at[log_idx].set(next_log['lat'][i]),
                'lon':log['lon'].at[log_idx].set(next_log['lon'][i]),
                'plan':log['plan'].at[log_idx].set(next_log['plan'][i]),}
            log_idx+=1 


    log = {
        't':log['t'][:log_idx],
        'h':log['h'][:log_idx],
        'lat':log['lat'][:log_idx],
        'lon':log['lon'][:log_idx],
        'plan':log['plan'][:log_idx]}

    return log

ELAPSED_TIME = 60*60*24*7

# print("Guess...")
# tplt.deterministic_plot_on_map(trajectory_at(START_TIME, balloon, make_plan(START_TIME, balloon, wind_inst, ELAPSED_TIME), wind_inst)[-1])

# # Get the optimal plan in the observed wind data, but then run it in the real wind field
# print("Without MPC...")
# optimal_plan_no_noise = get_optimal_plan(START_TIME, balloon, make_plan(START_TIME, balloon, wind_inst, ELAPSED_TIME), wind_inst) 
# tplt.deterministic_plot_on_map(trajectory_at(START_TIME, balloon, optimal_plan_no_noise, wind_inst)[-1])

# Runs receeding horizon control
print("Running MPC...")
 # 9 hours
logs = unjitted_receeding_horizon_control(START_TIME, ELAPSED_TIME, balloon, wind_inst, wind_inst)
tplt.deterministic_plot_on_map(logs)
