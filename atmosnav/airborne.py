from typing import Self
from .controllers import Controller
from .dynamics import Dynamics
import jax
from jax import Array
import jax.numpy as jnp

from .jaxtree import JaxTree

# StepType = tuple[Array, Controller, Dynamics]

class Airborne(JaxTree):
    """ 
    An agent represents any system affected by wind that is governed by dynamics and given inputs
    through actions.

    An action is defined by what the controller expects, which converts those actions into control inputs.
    Dynamics objects take control inputs and convert them into a change in agent state, which allows for
    the agent to be updated. 
    """
    
    def __init__(self, state: Array, controller: Controller, dynamics: Dynamics):
        """ Creates an agent defined by a controller and dynamics function at an initial state. 
        
        Args:
           state (jax.Array): an array representing the state, often a 1D vector
           controller (Controller): the controller of the agent
           dynamcis (Dynamics): the dynamics of the agent

        """
        self.state = state
        self.controller = controller
        self.dynamics = dynamics


    def step(self, time: jnp.float32, action: Array, wind_vector: Array) -> Self:
        """ Returns an updated agent at the next time step. 
        
        Args:
            time (jnp.float32): time of the system
            action (jax.Array): the action to take as a vector
            wind_vector (jax.Array): some wind disturbance as a vector

        Returns:
            (Agent): the updated agent
        """
        control_input, controller = self.controller.action_to_control_input(time, self.state, action)
        delta_state, dynamics = self.dynamics.control_input_to_delta_state(time, self.state, control_input, wind_vector)
        return Airborne(self.state + delta_state, controller, dynamics)
    
    def tree_flatten(self):
        return (self.state, self.controller, self.dynamics), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)