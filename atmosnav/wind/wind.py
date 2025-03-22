import abc
from ..jaxtree import JaxTree
import jax.numpy as jnp
# from jax import Array

Array = 'jax array'

class Wind(JaxTree):

    @abc.abstractmethod
    def get_direction(self, time: jnp.float32, state: Array) -> tuple[jnp.float32, jnp.float32]:
        """
            Gets the direction of the wind movement at the state and a given time.

            Args:
                time (jnp.float32): The time now since the simulation has started
                state (jnp.Array): The state vector, composed of [lat, lon, h]
            
            Returns:
                dv (jnp.float32): The direction along latitude
                du (jnp.float32): The diretcion along longitude
        """

