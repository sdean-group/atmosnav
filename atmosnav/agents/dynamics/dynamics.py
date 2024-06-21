from typing import Self

import abc
from ...jaxtree import JaxTree

import jax.numpy as jnp
from jax import Array

import jax

class Dynamics(JaxTree):
    """ 
    Computes a change in state of an agent based on a control input and wind vector. 
    
    Allows agents to be governed by physically-based dynamics.
    """

    @abc.abstractmethod
    def control_input_to_delta_state(self, time: jnp.float32, state: Array, control_input: Array, wind_vector: Array) -> tuple[Array, Self]:
        """
        Computes the change of the agent's state by taking in low-level control inputs 

        Args:
            time (jnp.float32): The time now since the simulation has started
            state (jax.Array): The state of the agent, often a one-dimensional jnp array
            control_input (jax.Array): The control input to the agent, often a one-dimensional jnp array
            wind_vector (jax.Array): The wind field

        Returns:
            (as a tuple: )
            delta_state (jax.Array): The change of state,
            Self (Dynamics): The updated Dynamics object.
        """