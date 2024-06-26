import jax
from jax import Array
import jax.numpy as jnp
from .jaxtree import JaxTree
from .controllers import *
from .dynamics import *
from .airborne import Airborne
from functools import partial 

class Arena(JaxTree):
    def __init__(self, start_time, airborne_systems, wind, waypoint_time_step, integration_time_step):
        self.start_time = start_time
        self.airborne_systems = airborne_systems
        self.wind = wind
        self.waypoint_time_step = waypoint_time_step
        self.integration_time_step = integration_time_step


    def get_trajectory(self, action_supplier):
        log = Log()
        while action_supplier.has_action():
            action = action_supplier.get_action()
            airborne = airborne.apply_action(action)
            log.add(airborne.state)
        return log
