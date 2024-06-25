from jax import lax
import jax.numpy as jnp

def p2alt(pressure):
    """ Converts air pressure to altitude."""
    return lax.cond(
        pressure > 22632.1,
        lambda p:44330.7 * (1 - (p / 101325.0) ** 0.190266) / 1000.0,
        lambda p: -6341.73 * jnp.log((0.176481 * p) / 22632.1) / 1000.0,
        operand=pressure
    )

def alt2p(altitude: float) -> float:
    """ Converts altitude to air pressure. """
    return lax.cond(
        altitude <= 11,
        lambda alt: lax.cond(
            alt < 0, 
            lambda _: 0.0, 
            lambda al: 3.83325e-20 * (44330.7 - al * 1000)**5.255799,
            operand=alt),
        lambda alt: 128241. *jnp.exp(-0.000157686 * alt * 1000),
        operand=altitude)

def bisect_left_jax(arr, x):
    """ A left binary search implemented in XLA """
    arr = jnp.asarray(arr)
    low, high = 0, len(arr)
    
    def cond_fun(state):
        low, high = state
        return low < high
    
    def body_fun(state):
        low, high = state
        mid = (low + high) // 2
        update = (arr[mid] < x)
        return (lax.select(update, mid + 1, low), lax.select(update, high, mid))
    
    low, _ = lax.while_loop(cond_fun, body_fun, (low, high))
    return low
