# from typing import Self
Self = 'Controller'

import abc
import jax.numpy as jnp
from ..jaxtree import JaxTree

# from jax import Array
import jax

Array = 'jax array'

class Controller(JaxTree):
    """ 
    Computes a control input given an action.
    
    Allows agents to be controlled using higher-level actions if their underlying dynamics are complicated. 
    """

    @abc.abstractmethod
    def action_to_control_input(self, time: jnp.float32, state: Array, action: Array) -> tuple[Array, Self]:
        """
        Computes the low-level control input from the higher-level action

        Args:
            time (jnp.float32): The time now since the simulation has started
            state (jax.Array): The state of the agent, often a one-dimensional jnp array
            action (jax.Array): The high-level action
        
        Returns:
            (as a tuple:)
            control_input (jax.Array): The low-level control input that will be directly applied to the agent
            Self (Controller): The updated Controller object
        """
