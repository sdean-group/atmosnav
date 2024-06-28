from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial 
from scipy.optimize import minimize
import time
import random

"""
This script runs multiple initializations of the gradient ascent algorithm to try and 
reveal the best initial plan
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
wind_inst= WindFromData.from_data(DATA_PATH, start_time=START_TIME, integration_time_step=INTEGRATION_TIME_STEP)

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

@jax.jit
def optimize_plan(start_time, balloon, plan, wind):
    def inner_opt(i, plan):
        d_plan = gradient_at(start_time, balloon, plan, wind)
        return plan + 0.5 * d_plan / jnp.linalg.norm(d_plan)
    return jax.lax.fori_loop(0, 83, inner_opt, init_val=plan)

def get_random_coefficients(x_min, x_max, L, U, num_terms):
    """
    Args:
        x_min (float): Min val of the domain
        x_max (float): Max val of the domain
        L (float): Lower bound of the range
        U (float): Upper bound of the range
        num_terms (int): Number of the polynomial terms
    
    Returns:
        A list of the coefficients.
    """

    def poly_func(x, coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))

    # Define the constraint function
    def constraint(coeffs, x, L, U):
        return max(0, L - poly_func(x, coeffs)) + max(0, poly_func(x, coeffs) - U)

    # Objective function for optimization
    def objective(coeffs, x_samples, L, U):
        return sum(constraint(coeffs, x, L, U) for x in x_samples)
    
    x_samples = np.linspace(x_min, x_max, num = 100)
    initial = [0] * num_terms
    result = minimize(objective, initial, args=(x_samples, L, U))
    return result.x.tolist()

def get_random_plan(num_points):
    L = random.randint(1, 21)
    U = random.randint(L, 21)
    num_terms = random.randint(1, 7)
    window_size = 2 * ( random.random() - 1) + 2

    coefficients = get_random_coefficients(0, 1, L, U, num_terms)
    f = lambda x: sum(coeff * (x**i) for i, coeff in enumerate(coefficients))
    
    x = np.linspace(0, 1, num = num_points)
    y = f(x)

    return np.vstack([ y + window_size/2, y - window_size/2 ]).T

WAYPOINT_COUNT = 40

def random_initializations(balloon, wind):
    logs = [] 
    for i in range(10):
        # print(i)
        balloon.dynamics = SimpleAltitudeModel()
        # AltitudeModel(integration_time_step=INTEGRATION_TIME_STEP, key=jax.random.key(time.time_ns()))
        # print('a')
        # uppers = 10 + jnp.sin(2*np.pi*jnp.arange(WAYPOINT_COUNT)/10)
        # lowers = uppers - 3
        # plan_to_optimize = np.vstack([lowers,uppers]).T
        plan_to_optimize = get_random_plan(WAYPOINT_COUNT)
        
        _, log = run(START_TIME, balloon, plan_to_optimize, wind)
        logs.append(log)
        
        # print('b')
        optimal_plan = optimize_plan(START_TIME, balloon, plan_to_optimize, wind)
        # print('c')

        _, log = run(START_TIME, balloon, optimal_plan, wind)
        logs.append(log)
        
        # print('d')

    return logs

logs = random_initializations(balloon, wind_inst)
for i in range(0, len(logs), 2):
    tplt.plot_on_map_many([logs[i], logs[i+1]], filename=f'randominit{i}.png')