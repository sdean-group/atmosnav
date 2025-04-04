from typing import Self
from atmosnav import Airborne, Controller, Dynamics
import jax
import jax.numpy as jnp
from jax import Array

"""
This tests the JIT compilation of pure (e.g. stateless) controller and dynamics objects in 
use with an agent.

"""

class PureController(Controller):

    def action_to_control_input(self, time: jnp.float32, state: Array, action: Array) -> tuple[Array, Self]:
        return (action * time * jnp.linalg.norm(state)), self
    
    def tree_flatten(self):
        return tuple(), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

class PureDynamics(Dynamics):

    def control_input_to_delta_state(self, time: jnp.float32, state: Array, control_input: Array, wind_vector: Array) -> tuple[Array, Self]:
        return jnp.sin(time) + control_input / (time+1), self
    
    def tree_flatten(self):
        return tuple(), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


agent = Airborne(jnp.array([ 0.0, 0.0, 0.0 ]), PureController(), PureDynamics())

@jax.jit
def fast_run(agent):
    def inner_loop(i, agent):
        jax.debug.print("{x}, {y}, {z}",x=agent.state[0], y=agent.state[1], z=agent.state[2])
        return agent.step(time=i, action=jnp.array([ 0.0 ]), wind_vector=jnp.array([0.0]))
    return jax.lax.fori_loop(0, 1000, inner_loop, init_val=agent)

fast_run(agent)