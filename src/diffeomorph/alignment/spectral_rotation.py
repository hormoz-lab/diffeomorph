from functools import partial

import jax
import jax.numpy as jnp

from .assemble_wigner_D import assemble_real_wigner_D_direct
from .s3_riemannian_optimizers import riemannian_adam_on_s3


def calculate_overlap(q, offset, binoms, prefactors, perms, ell_max, Z_pred, Z_target):
    wigner_Ds = assemble_real_wigner_D_direct(ell_max, q, offset, binoms, prefactors, perms)
    overlap = jax.vmap(
        lambda z_pred, z_target, wigner_D: jnp.sum(
            (z_pred @ wigner_D.T) * z_target),
        in_axes=(0, 0, 0),
    )(Z_pred[1:], Z_target[1:], wigner_Ds[0:]).sum()
    return -overlap


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 7))
def solve_inner_alignment_problem(lr, tol, q, offset, binoms, prefactors, perms, ell_max, Z_pred, Z_target):

    optimizer = riemannian_adam_on_s3(lr)
    opt_state = optimizer.init(q)
    
    init_state = (1e7, 1e4, q, opt_state)

    def cond_fn(state):
        prev_loss, curr_loss, *_,  = state
        return jnp.abs(curr_loss - prev_loss) > tol 

    def body_fn(state):
        _, curr_loss, q, opt_state = state
        
        new_loss, grads = jax.value_and_grad(
            calculate_overlap, 
        )(
            q, 
            offset,
            binoms,
            prefactors,
            perms,
            ell_max,
            Z_pred, Z_target,
        )    
        
        new_q, new_opt_state = optimizer.update(grads, opt_state, q)   
        return curr_loss, -new_loss, new_q, new_opt_state
    
    _, final_loss, final_q, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_q


# Forward and backward passes
def solve_inner_alignment_problem_fwd(lr, tol, q, offset, binoms, prefactors, perms, ell_max, Z_pred, Z_target):
    optimized_q = solve_inner_alignment_problem(lr, tol, q, offset, binoms, prefactors, perms, ell_max, Z_pred, Z_target)
    return optimized_q, (optimized_q, offset, binoms, prefactors, perms, Z_pred, Z_target)


def solve_inner_alignment_problem_bwd(lr, tol, ell_max, res, grad_vec):
    optimized_q, offset, binoms, prefactors, perms, Z_pred, Z_target = res
        
    d2O_dq2, d2O_dZdq = jax.jacobian(
        lambda _q, _Z_pred: 
            (jnp.eye(4) - jnp.outer(_q, _q)) @ 
            jax.grad(
                calculate_overlap, 
                argnums=0,
            )(_q, offset, binoms, prefactors, perms, ell_max, _Z_pred, Z_target),
        argnums=(0, 1),
    )(optimized_q, Z_pred)

    projector = jnp.eye(4) - jnp.outer(optimized_q, optimized_q)
    riemannain_hessian = projector @ d2O_dq2 @ projector
    inv_hessian = jnp.linalg.pinv(riemannain_hessian, rtol=2e-2) 
    dq_dZ = jnp.einsum(
        'i,ij,jk,knlm->nlm', 
        grad_vec, projector, -inv_hessian, d2O_dZdq,
    )
    return (None, None, None, None, None, dq_dZ, None)


solve_inner_alignment_problem.defvjp(
    solve_inner_alignment_problem_fwd, 
    solve_inner_alignment_problem_bwd,
)