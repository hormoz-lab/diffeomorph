from itertools import product
import math

import numpy as np

import jax.numpy as jnp


# Precompute normalization constants and prefactors
def precompute_factors(max_l):       
    offset = max_l
    prefactors = np.zeros((max_l+1, 2*max_l+1, 2*max_l+1)) 
    binoms = np.zeros((max_l+1, 2*max_l+1, 2*max_l+1, 2*max_l+1)) 
    
    for l in range(max_l + 1):
        for mp, m in product(range(-l, l+1), repeat=2):
            norm_const = math.sqrt(
                math.factorial(l + m) * math.factorial(l - m) /
                (math.factorial(l + mp) * math.factorial(l - mp))
            )
            prefactors[l, mp + offset, m + offset] = norm_const

            rho_min = max(0, mp - m)
            rho_max = min(l + mp, l - m)  
            for rho in range(rho_min, rho_max + 1):
                binomial_const = math.comb(l+mp, rho) * math.comb(l-mp, l-rho-m) * (-1)**rho
                binoms[l, mp + offset, m + offset, rho] = binomial_const
    
    return (prefactors, binoms)
  

# quaternionic algebra
def quaternion_conjugate(q):
    w, x, y, z = q
    return jnp.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    # Scalar-first: q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_from_a_to_b(a, b):
    """Compute quaternion that rotates vector 'a' onto vector 'b'."""
    c = jnp.dot(a, b)
    v = jnp.cross(a, b)
    s = jnp.linalg.norm(v)
    n = v / s
    theta = jnp.arctan2(s, c)
    half = 0.5*theta
    q = jnp.array([np.cos(half), *(np.sin(half) * n)])
    q /= jnp.linalg.norm(q)
    return q