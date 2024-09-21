from .dynamics import Dynamics
import jax
import jax.numpy as jnp
from jax import lax

from ..jaxtree import JaxTree
import math

Array = 'jax array' 

class SimpleAltitudeModel(Dynamics):
    def control_input_to_delta_state(self, time: jnp.float32, state: Array, control_input: Array, wind_vector: Array):
        new_h = jnp.sum(control_input)/2.0
        return jnp.array([ wind_vector[0], wind_vector[1], new_h - state[2], state[3]]), self


    def tree_flatten(self):
        return tuple(), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return SimpleAltitudeModel()


SinApproxT = 10 * 60


class SinApprox(JaxTree):
    def __init__(self, key):
        self.sin_mode = False
        self.y = 0.0
        self.x = 0.0
        self.leak = 0.0
        self.θ = 0.1
        self.m = 4.0
        self.vlim = 1.7
        self.max_dx = 1.0
        self.vgain = 1 / (60.0 * 20.0)
        self.key = key
    
    def make_copy(self):
        sin_approx = SinApprox(self.key)
        sin_approx.sin_mode = self.sin_mode
        sin_approx.y = self.y
        sin_approx.x= self.x
        sin_approx.leak = self.leak
        sin_approx.θ = self.θ
        sin_approx.m = self.m
        sin_approx.vlim = self.vlim
        sin_approx.max_dx = self.max_dx
        sin_approx.vgain = self.vgain
        return sin_approx
    
    def tree_flatten(self):
        children = (self.sin_mode, self.y, self.x, self.key)  # arrays / dynamic values
        aux_data = {'leak': self.leak, 
                    'θ': self.θ, 
                    'm': self.m, 
                    'vlim': self.vlim, 
                    'max_dx': self.max_dx, 
                    'vgain': self.vgain}  # static values
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        sin_approx = cls(children[-1])
        sin_approx.sin_mode, sin_approx.y, sin_approx.x = children[:-1]
        sin_approx.leak = aux_data['leak']
        sin_approx.θ = aux_data['θ']
        sin_approx.m = aux_data['m']
        sin_approx.vlim = aux_data['vlim']
        sin_approx.max_dx = aux_data['max_dx']
        sin_approx.vgain = aux_data['vgain']
        return sin_approx

class AltitudeModel(Dynamics):

    def __init__(self, integration_time_step, key):

        self.dt = integration_time_step
        self.sqrt_dt = math.sqrt(float(self.dt))
        self.s = SinApprox(key)
        
    def get_random_jax(self, s):
        rand = jax.random.uniform(s.key) - 0.5
        _, s.key = jax.random.split(s.key)
        return rand, s

    def control_input_to_delta_state(self, time: jnp.float32, state: Array, control_input: Array, wind_vector: Array):
        (h, v), next_am = self.update(state, control_input)
        return jnp.array([ wind_vector[0], wind_vector[1], h - state[2], v - state[3]]), next_am

    def update(self, state, waypoint):
        lbnd, ubnd = waypoint

        s = self.s.make_copy()

        def if_false(op):
            altitude, state, lbnd, ubnd, s = op
            z, s = altitude.get_random_jax(s)
            tol = ubnd - lbnd
            s.y = s.y * (1.0 - s.θ) + z * s.θ
            dx = (s.y * s.m) / tol
            s.x += dx * altitude.dt / SinApproxT

            s.x += lax.cond(s.leak != 0.0, 
                                     lambda op: -op[0] * jnp.cos(op[1]),
                                     lambda _: 0.0,
                                     operand=(s.leak, s.x))
            return (lbnd + (jnp.sin(s.x) + 1.0) * (tol / 2.0), state[3]), s
        
        def if_true(op):
            altitude, state, lbnd, ubnd, s = op
            cond, sin_approx = altitude.sin_state_init(s, state[2], state[3], lbnd, ubnd)
            
            return lax.cond(cond, 
                     lambda ops: if_false(ops),
                     lambda ops: ops[0].sin_vmode_update(ops[1], ops[2], ops[3], ops[4]),
                     operand = (altitude, state, lbnd, ubnd, sin_approx))

        h_and_v, self.s = lax.cond(jnp.invert(self.s.sin_mode),
                        if_true,
                        if_false, 
                        operand=(self, state, lbnd, ubnd, s))

        return h_and_v, self

    def _sin_state_init_valid_bounds(self, op):
        h, v, lbnd, ubnd, s = op 
        tol = ubnd - lbnd
        T = SinApproxT
        x = jnp.arcsin((h - lbnd) / (tol / 2.0) - 1.0)
        
        s_x = lax.cond(
            v > 0,
            lambda o: o,
            lambda o: math.pi - o,
            operand=x
        )
       
        inc = T * 2 * (v / 3600.0) / jnp.cos(s_x)
        s_y = inc / s.m

        sin_mode = jnp.invert((jnp.greater(jnp.abs((s_y * s.m) / tol), s.max_dx)))

        sa = s.make_copy()
        sa.x = s_x
        sa.y = s_y
        sa.sin_mode = sin_mode

        return sin_mode, sa

    def sin_state_init(self, s, h, v, lbnd, ubnd):
        return lax.cond(
            lbnd < h, 
            lambda op: lax.cond(op[0] < op[3],
                                self._sin_state_init_valid_bounds,
                                lambda _: (False, op[-1]),
                                operand = op),
            lambda op: (False, op[-1]),
            operand=(h, v, lbnd, ubnd, s))
        
    def sin_vmode_update(self, state, lbnd, ubnd, s):
        h, v, l, u = state[2], state[3], lbnd, ubnd
        v = jnp.select([h > u, h < l], 
                       [lax.cond(v > -s.vlim,
                                 lambda op: jnp.max(jnp.array([-op[1], op[0] + jnp.min(jnp.array([-op[1] - v, 0.0])) * op[2] * op[3]])),
                                 lambda op: op[0],
                                 operand=(v, s.vlim, s.vgain, self.dt)),
                                
                        lax.cond(v < s.vlim, 
                                lambda op: jnp.min(jnp.array([op[1], op[0] + jnp.max(jnp.array([op[1] - v, 0.0])) * op[2] * op[3]])),
                                lambda op: op[0],
                                operand=(v, s.vlim, s.vgain, self.dt))
                        ],
                       default = v + -v * s.vgain / 2.0 * self.dt)

        z, s = self.get_random_jax(s)
        v += 1.5 * self.sqrt_dt * 0.0408248 * z
        h += v / 3600.0 * self.dt
        return (h, v), s
    
    def tree_flatten(self):
        children = (self.s, )  # arrays / dynamic values
        aux_data = {'dt':self.dt, 'sqrt_dt':self.sqrt_dt}  # static values
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        am = cls(aux_data['dt'], children[0].key)
        am.s = children[0]
        return am