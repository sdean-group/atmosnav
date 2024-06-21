from typing import Self

import abc
from ...jaxtree import JaxTree

import jax.numpy as jnp
from jax import Array

import jax

class Dynamics(JaxTree):
    """ """

    @abc.abstractmethod
    def control_input_to_delta_state(self, time: jnp.float32, state: Array, control_input: Array, wind_vector: Array) -> tuple[Array, Self]:
        """ """