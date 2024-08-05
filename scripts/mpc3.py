from atmosnav import *
import atmosnav.trajplot as tplt
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial 
import optax
import matplotlib.pyplot as plt
import time

##################
# CONSTANTS    
##################

# The wind data directory
DATA_PATH = "../data/small"

# initial time (as unix timestamp), must exist within data
START_TIME = 1691366400

# the numerical integration time step for the balloon 
INTEGRATION_TIME_STEP = 60*10

# The time between waypoint
WAYPOINT_TIME_STEP = 60*60*3

# The weather balloon being tested
def make_weather_balloon(start, start_time, waypoint_time_step, integration_time_step):
    return Airborne(
        jnp.array([ start[0], start[1], start[2], 0.0 ]),
        PlanToWaypointController(start_time=start_time, waypoint_time_step=waypoint_time_step),
        DeterministicAltitudeModel(integration_time_step=integration_time_step))

# Possible start and end locations
LOCATIONS = {
    'ithaca': (42.4410187, -76.4910089, 0),
    'madrid': (40.4168, -3.7038, 0),
    'reykjavik': (64.1470, -21.9408, 0),
    'bar harbor': (44.3876, -68.2039, 0),
    'chicago': (41.8781, -87.6298, 0),
    'london': (51.5072, -0.1276, 0),
    'boston': (42.3601, -71.0589, 0),
    'palo alto': (37.4419, -122.1430, 0),
    'new york city': (40.7128, -74.0060, 0)
}

START = 'ithaca'
DESTINATION = 'reykjavik'

DISCOUNT_FACTOR, VERTICAL_WEIGHT = 0.99, 111
MAX_OPTIMIZER_STEPS = 100

ELAPSED_TIME, HORIZON_TIME, FOLLOW_TIME = 60*60*24*7, 60*60*24*3, 60*60*9

##################
# HELPER CODE    
##################


# Plan helper functions
#TODO: generate N plans, then calculate the cost all at once by vectorizing
# @partial(jax.jit, static_argnums=(4,)) # --> horizon time
def get_initial_plan(sim, start_time, balloon, wind, horizon_time, objective):
    num_plans = 100
    key = jax.random.key(time.time_ns())
    WAYPOINT_COUNT = horizon_time//WAYPOINT_TIME_STEP+1
    init_plan = jnp.zeros((WAYPOINT_COUNT, 1))
    init_cost = sim.cost_at(START_TIME, balloon, init_plan, wind, objective)

    def add_plan(i, key_bestplancost):
        key, best_plan, best_cost = key_bestplancost
        key, *subkeys = jax.random.split(key, 3)

        # TODO: maybe jax command to get the right size straight away
        plan = 22*jax.random.uniform(subkeys[0]) + jnp.sin(2*jnp.pi*jax.random.uniform(subkeys[1])*jnp.arange(WAYPOINT_COUNT)/10)
        plan = jnp.reshape(plan, (WAYPOINT_COUNT, 1))
        cost = sim.cost_at(start_time, balloon, plan, wind, objective)
        # TODO: replace with jnp.where
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

class FinalLongitude(Objective):
    def evaluate(self, times, states):
        return -states[-1][1]

    def tree_flatten(self): 
        return tuple(), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return FinalLongitude()

class FinalDistance(Objective):
    def __init__(self, destination):
        self.destination = jnp.array(destination)

    def evaluate(self, times, states):
        delta = (self.destination - states[-1][0:3])
        return delta @ delta

    def tree_flatten(self): 
        return (self.destination, ), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return FinalDistance(*children)
    
class DiscountedDistance(Objective):
    def __init__(self, destination, discount, vertical_weight):
        self.destination = jnp.array(destination)
        self.discount = discount
        self.vertical_weight = vertical_weight

    def evaluate(self, times, states):
        # TODO: implement discounted distance
        delta = (self.destination - states[-1][0:3])
        return delta @ delta

    def tree_flatten(self): 
        return (self.destination, self.discount, self.vertical_weight), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return DiscountedDistance(*children)

# TODO: make plan optimizer class? make this method reusable?
# TODO: replace this with optax optimizer
# @jax.jit
def optimize_plan(sim, start_time, balloon, plan, wind, steps, objective):
    def inner_opt(i, plan):
        d_plan = sim.gradient_at(start_time, balloon, plan, wind, objective)
        return plan - 0.5 * d_plan / jnp.linalg.norm(d_plan)
    return jax.lax.fori_loop(0, steps, inner_opt, init_val=plan)

def get_optimal_plan(sim, start_time, balloon, elapsed_time, wind, steps, objective):
    initial_plan = get_initial_plan(sim, start_time, balloon, wind, elapsed_time, objective)
    optimized_plan = optimize_plan(sim, start_time, balloon, initial_plan, wind, steps, objective)
    return optimized_plan
    

def mpc(sim: DifferentiableSimulator, start_time, balloon, true_wind, observed_wind, horizon_time, follow_time, elapsed_time, objective, steps_for_optimizer=100):
    time = start_time
    partial_logs = []

    while time < start_time + elapsed_time:
        print(time)
        optimal_plan = get_optimal_plan(sim, start_time, balloon, horizon_time, observed_wind, steps_for_optimizer, objective)
        plan_to_follow = optimal_plan[:follow_time//WAYPOINT_TIME_STEP+1]

        (time, balloon), log = sim.trajectory_at(time, balloon, plan_to_follow, true_wind)
        partial_logs.append(log)
        balloon.controller.start_time += follow_time # TODO: somehow make this step unnecessary

        # TODO: use previous optimal plan as apart of next plan's routine for picking for next initial plan

    keys = partial_logs[0].keys() # must have the same structure
    log = {key: np.concatenate([partial_log[key] for partial_log in partial_logs]) for key in keys}
    return (time, balloon), log


##################
# DRIVER CODE    
##################

if __name__ == "__main__":
    
    wind_inst = WindFromData.from_data(DATA_PATH, INTEGRATION_TIME_STEP)
    balloon = make_weather_balloon(LOCATIONS[START], START_TIME, WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP)

    sim = DifferentiableSimulator(WAYPOINT_TIME_STEP, INTEGRATION_TIME_STEP)
    obj = FinalDistance(LOCATIONS[DESTINATION], DISCOUNT_FACTOR, VERTICAL_WEIGHT)

    optimal_plan = get_optimal_plan(sim, START_TIME, balloon, ELAPSED_TIME, wind_inst, MAX_OPTIMIZER_STEPS, obj)
    _, log = sim.trajectory_at(START_TIME, balloon, optimal_plan, wind_inst)
    tplt.plot_on_map(log, filename='mpc3/nompc.png')

    _, log = mpc(sim, START_TIME, balloon, wind_inst, wind_inst, HORIZON_TIME, FOLLOW_TIME, ELAPSED_TIME, obj, MAX_OPTIMIZER_STEPS)
    tplt.plot_on_map(log, filename='mpc3/withmpc.png')