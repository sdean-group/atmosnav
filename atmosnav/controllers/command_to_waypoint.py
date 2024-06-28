
from .controller import Controller
from jax import Array
import jax.numpy as jnp
import jax

class CommandToWaypointController(Controller):

    def __init__(self, stay_epsilon = 0.5, altitude_delta = 0.1):
        self.stay_epsilon = stay_epsilon
        self.altitude_delta = altitude_delta



    def action_to_control_input(self, time: jnp.float32, state: Array, action: jnp.int32):
        h = state[2]
        stay_epsilon = self.stay_epsilon #km
        # if action is 0, delta is positive; if action is 2, delta is negative; if action is 1, delta is 0
        delta = -jnp.sign(action - 1) * self.altitude_delta
        # jax.debug.print("{a}, {delta}, {ep}, {h}", a=action, delta=delta, ep=stay_epsilon, h=h)
        return jnp.array([ h - stay_epsilon + delta, h + stay_epsilon + delta ]), self

        
        

    def tree_flatten(self): 
        return (self.stay_epsilon, self.altitude_delta), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        return CommandToWaypointController(*children)