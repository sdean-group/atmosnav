import equinox as eqx
import abc
import jax.numpy as jnp
import jax

# from jaxtyping import Array, Float, PyTree


class Dynamics(eqx.Module):

    @abc.abstractmethod
    def __call__(self, state, control, wind):
        """ """

    # consider n_dims: AbstractVar[int] with frozen=True for this
    @property
    @abc.abstractmethod
    def n_dims(self):
        """ """

class SimpleDynamics(Dynamics):
    
    x: float = 0.0

    @property
    def n_dims(self):
        return 2
    
    
    def __call__(self, state, control, wind):
        return jnp.array([ self.x, -self.x ]), SimpleDynamics(self.x + 1.0)


class DifferentiableSimulator(eqx.Module):

    steps: int = eqx.field(static=True)

    # IS IT NECESSARY TO USE A LOOP HERE?
    def trajectory_at(self, state, dynamics, steps):
        def update_state_once(carry, _):
            state, dynamics = carry
            d_state, dynamics = dynamics(state, jnp.zeros((1, )), jnp.zeros((2,)))
            state += 0.1 * d_state
            return (state, dynamics), (state, dynamics)

        initial_carry = (state, dynamics)
        # TODO: the stacking behavior from scan is interesting (though understably necessary for vectorization)
        return jax.lax.scan(update_state_once, initial_carry, None, length=steps)

state = jnp.zeros((2, ))
control = jnp.zeros((1, ))
wind = jnp.zeros((2, ))

dynamics = SimpleDynamics()
state_dot, dynamics = dynamics(state, control, wind)
print('d_state', state_dot)

sim = DifferentiableSimulator()
steps = 100
# print(jax.make_jaxpr(sim.trajectory_at)(state, dynamics, steps))
(next_state, dynamics), log = sim.trajectory_at(state, dynamics, steps)

print(next_state)
print(log)

trajectory_at = jax.jit(sim.trajectory_at, static_argnums=2)
(next_state, dynamics), log = trajectory_at(state, dynamics, steps)

print(next_state)
print(dynamics)

