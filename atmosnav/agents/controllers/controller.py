from typing import Self

import abc
import jax.numpy as jnp
from ...jaxtree import JaxTree

from jax import Array
import jax

class Controller(JaxTree):
    """ """

    @abc.abstractmethod
    def action_to_control_input(self, time: jnp.float64, state: Array, action: Array) -> tuple[Array, Self]:
        """  """
