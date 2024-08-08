from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np
import optax

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

# Load wind data
class WithDisturbance(Wind):
    def __init__(self, wind, key):
        self.wind = wind
        self.key = key

    def get_direction(self, time: jnp.float32, state: Array) -> tuple[jnp.float32, jnp.float32]:
        dv, du = self.wind.get_direction(time, state) 
        self.key, subkey = jax.random.split(self.key)
        dv_noise, du_noise = self.get_disturbance(subkey)
        return (dv*(1+dv_noise), du*(1+du_noise))

    def get_disturbance(self, key):
        return 0.25*jax.random.normal(key, (2,))
    
    def tree_flatten(self):
        return (self.wind, self.key), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return WithDisturbance(*children)
        
# Start and end
start = (42.4410187, -76.4910089, 0)

destinations = {"Bar Harbor":jnp.array([44.3876, -68.2039, 0]),
                "Reykjavik":jnp.array([64.1470, -21.9408, 0]),
                "Chicago":jnp.array([41.8781, -87.6298, 0]),
                "London":jnp.array([51.5072, -0.1276, 0]),
                }
#"Boston":jnp.array([42.3601, -71.0589, 0]),
# "Madrid":jnp.array([40.4168, -3.7038, 0]),


# Load wind data
wind_inst = WindFromData.from_data(DATA_PATH, start_time=START_TIME, integration_time_step=INTEGRATION_TIME_STEP)

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
def gradient_at(start_time, balloon, plan, wind, destination):
    N = (len(plan)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    def inner_run(i, time_balloon_cost):
        time, balloon, cost = time_balloon_cost
        cost += discount**(N-1-i)*((balloon.state[0]-destination[0])**2 + (balloon.state[1]-destination[1])**2 + ((balloon.state[2]-destination[2]))**2) 
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
def cost_at(start_time, balloon, plan, wind, destination):
    N = (len(plan)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    def inner_run(i, time_balloon_cost):
        time, balloon, cost = time_balloon_cost
        cost += discount**(N-1-i)*((balloon.state[0]-destination[0])**2 + (balloon.state[1]-destination[1])**2 + ((balloon.state[2]-destination[2]))**2) 
        # step the agent in time
        next_balloon, _ =balloon.step(time, plan, wind.get_direction(time, balloon.state))

        # jump dt
        next_time = time + INTEGRATION_TIME_STEP    

        return next_time, next_balloon, cost

    final_time, final_balloon, cost = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon, 0))
    #return -final_balloon.state[1]
    #return ((final_balloon.state[0]-destination[0])**2 + (final_balloon.state[1]-destination[1])**2 + (final_balloon.state[2])**2) 
    return cost


import time

# Create a plan
WAYPOINT_COUNT = 56 #  Total sim time = Waypoint Count * Waypoint Time Step = 40 * 3 hours = 5 days
num_plans = 5000
def make_plan(num_plans, WAYPOINT_COUNT, wind, destination):
        
    plans = [np.zeros((WAYPOINT_COUNT, 1))]

    for _ in range(num_plans):
        plan = 22*np.random.rand(1) + jnp.sin(2*np.pi*np.random.rand(1)*np.arange(WAYPOINT_COUNT)/10)
        plans.append(np.reshape(plan, (WAYPOINT_COUNT, 1)))

    best_plan = -1
    best_cost = np.inf
    for i, plan in enumerate(plans):
        
        cost = cost_at(START_TIME, balloon, plan, wind, destination)
        if cost < best_cost:
            best_plan = i
            best_cost = cost

    plan = plans[best_plan]
    return plan

# _, log = run(START_TIME, balloon, plan, wind)
# # tplt.plot_on_map(log)
# tplt.deterministic_plot_on_map(log)

start = time.time()

def get_optimal_plan(balloon, plan, wind, destination):
    optimizer = optax.adam(learning_rate = 0.03)
    opt_state = optimizer.init(plan)
    grad = gradient_at(START_TIME, balloon, plan, wind, destination)

    for i in range(10000):
        updates, opt_state = optimizer.update(grad, opt_state, plan)
        plan = optax.apply_updates(plan, updates)
        grad = gradient_at(START_TIME, balloon, plan, wind, destination)
    
    return plan

def optimized_test(balloon, observed_wind, true_wind, destination):
    plan = make_plan(5000, WAYPOINT_COUNT, observed_wind, destination)
    plan = get_optimal_plan(balloon, plan, observed_wind, destination)
    return run(START_TIME, balloon, plan, true_wind)


#print(f'Took: {time.time() - start} s')

import matplotlib.pyplot as plt

for destination in destinations.keys():
    print(destination)
    plt.figure(figsize=(10,6))
    ax1 = tplt.make_map_axis(ncol=2, nrow=1, pos=1)
    for i in range(10):
        key=jax.random.key(i)
        end_balloon, log = optimized_test(balloon, wind_inst, WithDisturbance(wind_inst, key), destinations[destination])
        ax1.plot(log['lon'],log['lat'])
        print((end_balloon.state[0]-destinations[destination][0])**2 + (end_balloon.state[1]-destinations[destination][1])**2 + ((end_balloon.state[2]-destinations[destination][2])/111)**2)
    plt.show()
# tplt.plot_on_map(log)
#tplt.deterministic_plot_on_map(log)