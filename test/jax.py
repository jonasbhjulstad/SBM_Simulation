import numpy as np
import jax.numpy as jnp
import jax

#create an example function to be jit compiled
@jax.jit
def f(x):
    return jnp.sin(x)

if __name__ == '__main__':
    #get jit code for f
    f_jit = jax.xla_computation(f)(jnp.array(0.0))
    #get the hlo code'
    print(f_jit.as_hlo_text())
    