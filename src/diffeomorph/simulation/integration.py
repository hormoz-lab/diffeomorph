import numpy as np
import jax.numpy as jnp
import jax.random as jr

from diffrax import (
    diffeqsolve, 
    MultiTerm,
    ODETerm,
    ControlTerm,
    VirtualBrownianTree, 
    SaveAt,
    Euler,
)
import lineax as lx


def integrate_sde(
    force_model, 
    x0,
    key,
    solver,
    epsilons,
    num_snapshots,
    t0,
    t1,
    dt0,
):
    keys = jr.split(key)

    def drift(t, y, args): 
        """Returns a drift(t, y, args) that generates a new key every call."""
        step_idx = jnp.floor(t / dt0).astype(jnp.int32)
        subkey   = jr.fold_in(keys[0], step_idx)
        return force_model(t, y, subkey, True)

    def diffusion(t, y, args):
        return lx.DiagonalLinearOperator(epsilons)
    
    brownian_motion = VirtualBrownianTree(
        t0, t1, tol=1e-3, shape=x0.shape, key=keys[1],
    )   

    terms = MultiTerm(
        ODETerm(drift),
        ControlTerm(diffusion, brownian_motion),
    )

    saveat = SaveAt(ts=np.linspace(t0, t1, num_snapshots))

    sol = diffeqsolve(
        terms=terms,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=x0,
        saveat=saveat,
    )
    
    return sol


class Euler_with_proj(Euler):
    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        y1, err, dense_info, new_solver_state, result = \
            super().step(terms, t0, t1, y0, args, solver_state, made_jump)
        y1_dir = y1[:, -3:]

        y1_dir_norm = jnp.sqrt(jnp.sum(y1_dir**2, axis=1, keepdims=True) + 1e-11)
        y1_dir /= y1_dir_norm
        y1 = jnp.concatenate((y1[:, :-3], y1_dir), axis=1)
        return y1, err, dense_info, new_solver_state, result