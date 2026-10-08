#!/usr/bin/env python3
"""Convolutional spiking feature extractor (CSNN) for the event camera, with the forward ToF distance fused into
its last layer; trained offline and used frozen
(weights in frozen_csnn_weights.msgpack)."""

import jax
import jax.numpy as jnp
import flax.linen as nn

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sparse_forest import N_PIXELS

# ============================================================================
# 1. Surrogate Gradient
# ============================================================================
@jax.custom_vjp
def surrogate_spike(v, v_th=1.0):
    return jnp.where(v >= v_th, 1.0, 0.0)

def spike_fwd(v, v_th):
    return surrogate_spike(v, v_th), (v, v_th)

def spike_bwd(res, g):
    v, v_th = res
    alpha = 10.0 
    sg_grad = alpha / (1.0 + jnp.abs(alpha * (v - v_th)))**2
    return (g * sg_grad, None)

surrogate_spike.defvjp(spike_fwd, spike_bwd)

# ============================================================================
# 2. Frozen CSNN Base
# ============================================================================
class CSNN_Base(nn.Module):
    """
    Maps the ON/OFF event time surface to a 256-dim feature vector; ToF is concatenated after the
    convolutions (late fusion).
    """
    @nn.compact
    def __call__(self, time_surface, x_tof):
        batch_size = time_surface.shape[0]
        
        # Split the concatenated ON/OFF surface and stack the two polarities as channels of the same pixel
        on_ts = time_surface[:, :N_PIXELS]
        off_ts = time_surface[:, N_PIXELS:]
        
        # (B, N_PIXELS, 2)
        x_vision = jnp.stack([on_ts, off_ts], axis=-1)

        # 1D Convolutions
        v1 = nn.Conv(features=16, kernel_size=(5,), strides=(2,), padding='SAME')(x_vision)
        s1 = surrogate_spike(v1, v_th=0.1)

        v2 = nn.Conv(features=32, kernel_size=(5,), strides=(2,), padding='SAME')(s1)
        s2 = surrogate_spike(v2, v_th=0.1)

        barcode = s2.reshape((batch_size, -1)) 

        # Late Fusion for ToF
        tof_flat = x_tof.reshape((batch_size, -1))
        fused_barcode = jnp.concatenate([barcode, tof_flat], axis=-1)

        # 256 features
        v_out = nn.Dense(features=256)(fused_barcode)
        
        # Positive firing rates via softplus (the activation used in training)
        return nn.softplus(v_out)