from typing import Self
from .controllers import Controller
from .dynamics import Dynamics
import jax
from jax import Array
import jax.numpy as jnp

from ..jaxtree import JaxTree

# StepType = tuple[Array, Controller, Dynamics]

class Agent(JaxTree):
    """ """
    
    def __init__(self, state: Array, controller: Controller, dynamics: Dynamics):
        """ """
        super().__init__()
        self.state = state
        self.controller = controller
        self.dynamics = dynamics


    def step(self, time: jnp.float32, action: Array, wind_vector: Array) -> Self:
        control_input, controller = self.controller.action_to_control_input(time, self.state, action)
        delta_state, dynamics = self.dynamics.control_input_to_delta_state(time, self.state, control_input, wind_vector)
        return Agent(self.state + delta_state, controller, dynamics)
    
    def tree_flatten(self):
        return (self.state, self.controller, self.dynamics), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)