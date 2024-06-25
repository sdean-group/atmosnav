
from .controller import Controller
from jax import Array
import jax.numpy as jnp

def lerp(a, b, t):
    return a + t * (b - a)

class PlanToWaypointController(Controller):

    def __init__(self, waypoint_time_step):
        self.waypoint_time_step = waypoint_time_step # this dt is how often a new waypoint happens (e.g. set a waypoint every 3 hours hours)

    def action_to_control_input(self, time: jnp.float32, state: Array, action: Array):
        idx = time // self.waypoint_time_step
        theta = (time - idx * self.waypoint_time_step) / float(self.waypoint_time_step)
        return lerp(action[idx], action[(idx+1)], theta), self
    
    def tree_flatten(self):
        children = (self.start_time, )  # arrays / dynamic values
        aux_data = {'waypoint_time_step': self.waypoint_time_step}  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return PlanToWaypointController(children[0], aux_data['waypoint_time_step'])

# class WaypointToWaypointController()
