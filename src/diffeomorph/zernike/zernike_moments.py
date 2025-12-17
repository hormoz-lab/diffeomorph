from .radial import *

import jax
import jax.numpy as jnp

from e3nn_jax import IrrepsArray, spherical_harmonics


def constrain_on_unit_ball(X, alpha, r_true_max=None):   
    def _differentiable_max(x, alpha):
        return (1/alpha) * jnp.log(jnp.sum(jnp.exp(alpha * x)))
    
    center = X.mean(axis=0, keepdims=True)
    X -= center

    if r_true_max is not None:
        X /= r_true_max
    else:
        r = jnp.sqrt(jnp.sum(X**2, -1))
        X /= _differentiable_max(r, alpha)
        
    return X


def calculate_zernike_moments(
    l_max, 
    ls, 
    irreps_out, 
    X,
    weights,
):
    radial_basis_functions = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10]

    r = jnp.sqrt(jnp.sum(X.array**2, -1, keepdims=True))

    # radial part
    R = jax.vmap(
        jax.lax.switch, 
        in_axes=(0, None, None),
    )(
        ls, 
        radial_basis_functions, 
        r,
    )

    # angular part
    Y = spherical_harmonics(irreps_out, X, True, 'integral')

    # differential volume element
    jacobian_factor = 1.
    
    #Z = []
    Z_lnm = jnp.zeros((l_max+1, 11, 21))
    for l, (R_n, n_cutoff, start, size) in enumerate(zip(
        R, 
        [11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6],
        [0,  1,  4,  9, 16, 25, 36, 49, 64, 81, 100], 
        [1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21],
    )):
        Z_nm = jnp.einsum(
            'N,Nn,Nm->nm',
            weights,
            R_n[:, :n_cutoff] * jacobian_factor, 
            Y.array[:, start:start+size],
        ) / R_n.shape[0]
        #Z.append(Z_nm)
        Z_lnm = Z_lnm.at[l, :n_cutoff, :size].set(Z_nm)

    return Z_lnm


def project_to_zernike_moments(X, weights, l_max, ls, irreps_out, alpha, r_true_max=None):
    if r_true_max is not None:
        X = constrain_on_unit_ball(X, alpha, r_true_max)
    else:
        X = constrain_on_unit_ball(X, alpha)
    X = IrrepsArray('1o', X)
    Z = calculate_zernike_moments(l_max, ls, irreps_out, X, weights)
    return Z
