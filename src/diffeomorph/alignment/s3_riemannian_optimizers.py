from typing import NamedTuple

import jax.numpy as jnp
from jax.lax import cond
import optax


# helpers
def extract_tangential_component(q, v):
    return v - jnp.dot(v, q) * q


def exponential_map(q, v, lr):
    norm_v = jnp.linalg.norm(v)
    return jnp.cos(lr * norm_v) * q - jnp.sin(lr * norm_v) * v / norm_v


def parallel_transport_sn(q, p, v):
    dot_qp = jnp.dot(q, p)
    dot_vp = jnp.dot(v, p)
    denom = 1.0 + dot_qp
    return v - (dot_vp / denom + 1e-8) * (q + p)

# SGD
class RiemannianSGDState(NamedTuple):
    count: jnp.ndarray


def riemannian_sgd_on_s3(lr: float) -> optax.GradientTransformation:

    def init_fn(q):
        return RiemannianSGDState(count=jnp.zeros([], dtype=jnp.int32))

    def update_fn(g, state, q):
        count = state.count + 1

        g_tangent = extract_tangential_component(q, g)
        q_new = exponential_map(q, g_tangent, lr)
        q_new = q_new / jnp.linalg.norm(q_new)

        return q_new, RiemannianSGDState(count=count)

    return optax.GradientTransformation(init=init_fn, update=update_fn)


# ADAM
class RiemannianAdamState(NamedTuple):
    count: jnp.ndarray
    m: jnp.ndarray
    v: jnp.ndarray
    prev_q: jnp.ndarray


def riemannian_adam_on_s3(
    lr=1e-3, b1=0.9, b2=0.999, eps=1e-8
) -> optax.GradientTransformation:

    def init_fn(q):
        return RiemannianAdamState(
            count=jnp.zeros([], dtype=jnp.int32),
            m=jnp.zeros_like(q),
            v=jnp.zeros_like(q),
            prev_q=q,  
        )

    def update_fn(g, state, q):
        count = state.count + 1
        g_tangent = extract_tangential_component(q, g)

        previous_m, previous_v = cond(
            state.count > 1,
            lambda _: (
                parallel_transport_sn(state.prev_q, q, state.m),
                parallel_transport_sn(state.prev_q, q, state.v),
            ),
            lambda _: (state.m, state.v),
            operand=None,
        )

        m = b1 * state.m + (1 - b1) * g_tangent
        v = b2 * state.v + (1 - b2) * (g_tangent ** 2)

        m_hat = m / (1 - b1 ** count)
        v_hat = v / (1 - b2 ** count)

        update = m_hat / (jnp.sqrt(v_hat) + eps)
        update = extract_tangential_component(q, update)  # ensure tangent

        q_new = exponential_map(q, update, lr)
        q_new = q_new / jnp.linalg.norm(q_new)

        new_state = RiemannianAdamState(count, m, v, q)
        return q_new, new_state

    return optax.GradientTransformation(init=init_fn, update=update_fn)