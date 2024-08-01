from .jaxtree import JaxTree
import jax
import jax.numpy as jnp
import abc

class Objective(JaxTree): # TODO: move to another file with a corresponding folder for objective functions
    
    @abc.abstractmethod
    def evaluate(self, log):
        pass

class DifferentiableSimulator(JaxTree):
    
    def __init__(self, waypoint_time_step, integration_time_step):
        self.waypoint_time_step = waypoint_time_step
        self.integration_time_step = integration_time_step
    
    @jax.jit
    def trajectory_at(self, start_time, airborne, plan, wind):
        N = (len(plan)-1)*self.waypoint_time_step//self.integration_time_step
        log = {
            't': jnp.zeros((N, ), dtype=jnp.int32),
            'h': jnp.zeros((N, )), 
            'lat': jnp.zeros((N, )), 
            'lon': jnp.zeros((N, )),
            'lbnd': jnp.zeros((N, )),
            'ubnd': jnp.zeros((N, ))}

        def inner_run(i, time_and_balloon_and_log):
            time, balloon, log = time_and_balloon_and_log

            # step the agent in time
            next_balloon, info = balloon.step(time, plan, wind.get_direction(time, balloon.state))
            
            # update the log
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

    def cost_at(self, start_time, airborne, plan, wind, objective):
        N = (len(plan)-1)*self.waypoint_time_step//self.integration_time_step

        def inner_run(i, time_and_balloon):
            time, balloon = time_and_balloon

            # step the agent in time
            next_balloon, info = balloon.step(time, plan, wind.get_direction(time, balloon.state))
            
            # jump dt
            next_time = time + self.integration_time_step

            return next_time, next_balloon
        
        final_time, final_balloon = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, airborne))
        return final_balloon.state[1]

    @jax.jit
    def gradient_at(self, start_time, airborne, plan, wind, objective):
        return jax.grad(self.cost_at, argnums=2)(start_time, airborne, plan, wind, objective)

    def tree_flatten(self): 
        return (), {'waypoint_time_step': self.waypoint_time_step, 'integration_time_step': self.integration_time_step}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        return DifferentiableSimulator(**aux_data)

