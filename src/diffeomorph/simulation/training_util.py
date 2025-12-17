import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

from .integration import integrate_sde
from .loss import shape_matching_loss
from diffeomorph.zernike import project_to_zernike_moments


def train_step(
    model, 
    optimizer,
    opt_state, 
    x0, 
    q, 
    Z_targets,
    key, 
    params,
    weight_index = None,
):

    #def _outer_loss(model):
    def _outer_loss(model, x0, q, Z_targets, key, params, weight_index):

        sol = integrate_sde(model, x0, key, **params['integration'])
        pred = sol.ys[-1]
        pred_X = pred[:, :3]      # coord
        pred_W = pred[:, 3:-3]    # gene

        uniform_weights = jnp.ones(pred_X.shape[0])
        Z_pred_unweighted = project_to_zernike_moments(
            pred_X, uniform_weights, **params['spectral'],
        )

        if weight_index is not None:
            Z_pred_weighted = project_to_zernike_moments(
                pred_X, pred_W[:, weight_index], **params['spectral'],
            )
            use_weighted = True
        else:
            Z_pred_weighted = None
            use_weighted = False

        Z_preds = {
            'unweighted': Z_pred_unweighted,
            'weighted': Z_pred_weighted,
        }

        spectral_loss, optimized_q = shape_matching_loss(
            q, Z_preds, Z_targets, use_weighted, **params['wigner'])

        com_loss = jnp.sum(pred_X.mean(axis=0)**2)

        return spectral_loss + com_loss, (optimized_q, sol.ys)

    #(loss_val, (optimized_q, traj)), grads = eqx.filter_value_and_grad(_outer_loss, has_aux=True)(model)
    (loss_val, (optimized_q, traj)), grads = eqx.filter_value_and_grad(
        _outer_loss, has_aux=True
    )(model, x0, q, Z_targets, key, params, weight_index) 

    updates, opt_state = optimizer.update(
        grads, 
        opt_state, 
        eqx.filter(model, eqx.is_array),
    )
    
    model = eqx.apply_updates(model, updates)

    return loss_val, model, opt_state, optimized_q, traj


def evaluate(
    models,  
    x0, 
    q,
    Z_targets, 
    params,
    weight_index = None,
):
    
    sol = integrate_sde(model, x0, **params['integration'])
    pred = sol.ys[-1]
    pred_X = pred[:, :3]               # coord
    pred_W = pred[:, 3:-3]             # gene

    uniform_weights = jnp.ones(pred_X.shape[0])
    Z_pred_unweighted = project_to_zernike_moments(
        pred_X, uniform_weights, **params['spectral'],
    )

    if weight_index is not None:
        Z_pred_weighted = project_to_zernike_moments(
            pred_X, pred_W[:, weight_index], **params['spectral'],
        )
        use_weighted = True
    else:
        Z_pred_weighted = None
        use_weighted = False

    Z_preds = {
        'unweighted': Z_pred_unweighted,
        'weighted': Z_pred_weighted,
    }

    spectral_loss, _ = shape_matching_loss(
        q, Z_preds, Z_targets, use_weighted, **params['wigner'])

    com_loss = jnp.sum(pred_X.mean(axis=0)**2)

    return spectral_loss + com_loss, sol.ys


def perturb_positions(tensor, noise_mag, key):
    position_noise = noise_mag * jr.normal(key, tensor[:, :3].shape)
    return tensor.at[..., :3].add(position_noise)