from typing import Self
from atmosnav import Airborne, Controller, Dynamics
import jax
import jax.numpy as jnp
from jax import Array

"""
This tests the JIT compilation of controller and dynamics objects with state in 
use with an agent.

This test also features logging to ensure that generating a trajectory from an agent is possible.

"""

class StatefulController(Controller):
    
    def __init__(self, x=1.0, y=2.0 ,z=3.0):
        self.x = x
        self.y = y
        self.z = z

    def action_to_control_input(self, time: jnp.float32, state: Array, action: Array) -> tuple[Array, Self]:
        dx = state[0] - self.x
        return (action * time * jnp.sin(jnp.linalg.norm(state.at[0].set(dx)))), StatefulController(dx, self.y, self.z)
    
    def tree_flatten(self):
        return (self.x, self.y, self.z), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class StatefulDynamics(Dynamics):

    def __init__(self, x=1.0):
        self.x = x

    def control_input_to_delta_state(self, time: jnp.float32, state: Array, control_input: Array, wind_vector: Array) -> tuple[Array, Self]:
        x = self.x * self.x
        return (control_input * x), StatefulDynamics(x)
    
    def tree_flatten(self):
        return (self.x, ), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


agent = Airborne(jnp.array([ 3.0, 2.0, 1.0 ]), StatefulController(), StatefulDynamics())

@jax.jit
def fast_run_with_log(agent):
    N = 1000
    log = {'x':jnp.zeros((N, )), 'y': jnp.zeros((N, )), 'z': jnp.zeros((N, ))}
    def inner_loop(i, agent_and_log):
        agent, log = agent_and_log
        jax.debug.print("{x}",x=i)
        next_log = {'x':log['x'].at[i].set(agent.state[0]),
               'y':log['y'].at[i].set(agent.state[1]),
               'z':log['z'].at[i].set(agent.state[2]),}
        next_agent = agent.step(time=i, action=jnp.array([ 0.0 ]), wind_vector=jnp.array([0.0]))
        return next_agent, next_log

    return jax.lax.fori_loop(0, N, inner_loop, init_val=(agent, log))

agent, log = fast_run_with_log(agent)
print(log)