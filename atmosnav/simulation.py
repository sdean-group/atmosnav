from .jaxtree import JaxTree
import jax
import jax.numpy as jnp
import abc
from line_profiler import profile

class Objective(JaxTree): # TODO: move to another file with a corresponding folder for objective functions
    
    @abc.abstractmethod
    def evaluate(self, times, states):
        """ 
        Calculates a score for a plan given the states it went to. 
        
        Because it calculates this score after the fact, an objective has access to all data and once
        and will usually not need to store state (also meaning it only has to return the score, and not the
        'next' objective)
        
        Could be extended to also contain the actions taken.
        """

class DifferentiableSimulator(JaxTree):
    
    def __init__(self, waypoint_time_step, integration_time_step):
        self.waypoint_time_step = waypoint_time_step
        self.integration_time_step = integration_time_step
    
    @jax.jit
    @profile
    def trajectory_at(self, start_time, airborne, plan, wind):
        N = (len(plan)-1)*self.waypoint_time_step//self.integration_time_step
        log = {
            't': jnp.zeros((N, ), dtype=jnp.int32),
            'h': jnp.zeros((N, )), 
            'lat': jnp.zeros((N, )), 
            'lon': jnp.zeros((N, )),
            'lbnd': jnp.zeros((N, )),
            'ubnd': jnp.zeros((N, ))}


        @profile
        def inner_run(i, time_and_balloon_and_log):
            time, balloon, log = time_and_balloon_and_log

            # step the agent in time
            next_balloon, info = balloon.step(time, plan, wind.get_direction(time, balloon.state))
            
            # update the log
            # Consider using jax.lax.scan for the log instead, may be more efficient
            next_log = {
                't':log['t'].at[i].set(time.astype(int)),
                'h':log['h'].at[i].set(balloon.state[2]),
                'lat':log['lat'].at[i].set(balloon.state[0]),
                'lon':log['lon'].at[i].set(balloon.state[1]),
                'lbnd':log['lbnd'].at[i].set(info['control_input'][0]),
                'ubnd':log['ubnd'].at[i].set(info['control_input'][1])}
            
            # jump dt
            next_time = time + self.integration_time_step

            return next_time, next_balloon, next_log
        
        final_time, final_balloon, log = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, airborne, log))
        return (final_time, final_balloon), log

    @jax.jit
    def cost_at(self, start_time, airborne, plan, wind, objective):
        N = (len(plan)-1)*self.waypoint_time_step//self.integration_time_step

        def inner_run(time_and_balloon, _):
            time, balloon = time_and_balloon

            # step the agent in time
            next_balloon, info = balloon.step(time, plan, wind.get_direction(time, balloon.state))
            
            # jump dt
            next_time = time + self.integration_time_step

            carry = next_time, next_balloon
            return carry, (next_time, next_balloon.state)
        
        (final_time, final_balloon), log = jax.lax.scan(inner_run, init=(start_time, airborne), xs=None, length=N)
        return objective.evaluate(log[0], log[1])
        

    @jax.jit
    def gradient_at(self, start_time, airborne, plan, wind, objective):
        return jax.grad(self.cost_at, argnums=2)(start_time, airborne, plan, wind, objective)

    def tree_flatten(self):
        return (), {'waypoint_time_step': self.waypoint_time_step, 'integration_time_step': self.integration_time_step}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        return DifferentiableSimulator(**aux_data)

