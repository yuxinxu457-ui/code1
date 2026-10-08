#!/usr/bin/env python3
"""Adaptive event-camera features learned online by unsupervised STDP (the plastic stream of the vision
front end), written as pure functions of an explicit state tuple."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import NamedTuple

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sparse_forest import N_PIXELS, DT

# ============================================================================
# 1. STDP Hyperparameters
# ============================================================================

N_INPUT = N_PIXELS          
N_HIDDEN = 256              
N_TOF_CHANNELS = 4          

BETA_LIF = 0.85             
V_TH_STDP = 1.0             

DT_MS = DT * 1000           
A_PLUS = 0.01               
A_MINUS = 0.012             
W_MAX = 0.15                
W_MIN = 0.0                 
W_INIT_MEAN = 0.05          
W_INIT_STD = 0.02        

TAU_TRACE = 100.0       # eligibility-trace decay time constant (ms); spans several frames
TRACE_DECAY = np.exp(-DT_MS / TAU_TRACE)

K_WTA = 12                  

THETA_INC = 0.50              
THETA_DECAY = 0.986          
THETA_MAX = 5.0              

# ============================================================================
# 2. State tuple
# ============================================================================
class VisionSTDPState(NamedTuple):
    W: jnp.ndarray
    e_pre: jnp.ndarray
    e_post: jnp.ndarray
    v_hidden: jnp.ndarray
    v_th_adapt: jnp.ndarray
    spike_trace: jnp.ndarray  # decaying spike count

# ============================================================================
# 3. Vision STDP Network Architecture
# ============================================================================
def lif_step(v, i_ext, beta=BETA_LIF):
    """Discrete leaky integrate-and-fire membrane update (integration and leak only)."""
    return beta * v + i_ext

def localized_inhibition(spike, pool_size=16, k_per_pool=1):
    """Retinotopic block-local winner-take-all."""
    B, N = spike.shape
    n_pools = N // pool_size
    
    spike_pools = spike.reshape((B, n_pools, pool_size))
    _, topk_idx = jax.lax.top_k(spike_pools, k_per_pool)
    
    batch_idx = jnp.arange(B, dtype=jnp.int32)[:, None, None]
    pool_idx = jnp.arange(n_pools, dtype=jnp.int32)[None, :, None]
    
    winner_mask = jnp.zeros((B, n_pools, pool_size), dtype=jnp.bool_)
    winner_mask = winner_mask.at[batch_idx, pool_idx, topk_idx].set(True)
    winner_mask_flat = winner_mask.reshape((B, N)).astype(jnp.float32)

    # pools without any spike produce no winner
    return jnp.where(spike > 0.0, winner_mask_flat, 0.0)

class STDPLayer:
    def __init__(self, n_pre, n_post, eta=A_PLUS, gamma=A_MINUS / A_PLUS, w_min=W_MIN, w_max=W_MAX):
        self.n_pre = n_pre
        self.n_post = n_post
        self.eta = eta
        self.gamma = gamma
        self.w_min = w_min
        self.w_max = w_max

        # Retinotopic receptive-field mask
        n_pixels = n_pre // 2
        neurons_per_pixel = n_post / n_pixels
        neuron_idx = jnp.arange(n_post)
        pixel_idx = jnp.arange(n_pixels)
        
        center_pixels = neuron_idx / neurons_per_pixel
        dist = jnp.abs(center_pixels[:, None] - pixel_idx[None, :])
        dist_pol = jnp.tile(dist, (1, 2))
        
        RF_RADIUS = 8.0 
        self.mask = (dist_pol < RF_RADIUS).astype(jnp.float32)

    def init_weights(self, key):
        W = random.normal(key, (self.n_post, self.n_pre), dtype=jnp.float32) * W_INIT_STD + W_INIT_MEAN
        return jnp.clip(W, self.w_min, self.w_max) * self.mask

    def apply_batch(self, W, e_pre, e_post, spike_pre_all, spike_post_all):
        dW_batch = (self.eta * jnp.einsum('bp,bo->bop', e_pre, spike_post_all)
                     - self.eta * self.gamma * jnp.einsum('bo,bp->bop', e_post, spike_pre_all))

        new_e_pre = TRACE_DECAY * e_pre + spike_pre_all
        new_e_post = TRACE_DECAY * e_post + spike_post_all

        dW = dW_batch.mean(axis=0) 
        bound_factor = 1.0 - (W - self.w_min) / (self.w_max - self.w_min + 1e-8)
        dW_pos = dW * jnp.clip(bound_factor, 0.05, 1.0)
        dW_neg = dW * jnp.clip(1.0 - bound_factor, 0.05, 1.0)
        dW = jnp.where(dW >= 0, dW_pos, dW_neg)

        # Homeostatic synaptic scaling: the total incoming weight of each hidden neuron is held constant
        # (L1 normalisation), which keeps weights stable while idle and makes new features compete.
        target_sum = jnp.sum(self.mask, axis=-1, keepdims=True) * W_INIT_MEAN
        
        # Add the learning updates on top of the current weights
        new_W = jnp.clip(W + dW, self.w_min, self.w_max) * self.mask
        
        # Apply multiplicative scaling to keep total input synaptic strength normalized
        current_sum = jnp.sum(new_W, axis=-1, keepdims=True)
        scale_factor = target_sum / (current_sum + 1e-8)
        new_W = jnp.clip(new_W * scale_factor, self.w_min, self.w_max) * self.mask
        
        return new_W, new_e_pre, new_e_post


class VisionSTDP:
    def __init__(self, key, n_input=N_INPUT, n_hidden=N_HIDDEN, k_wta=K_WTA, tof_channels=N_TOF_CHANNELS):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.k_wta = k_wta
        self.tof_channels = tof_channels
        self.n_polarized = 2 * n_input
        self.key = key
        
        self.stdp_layer = STDPLayer(self.n_polarized, n_hidden)

    def init_state(self, B) -> VisionSTDPState:
        """Initial state for a batch of B streams."""
        W_init = self.stdp_layer.init_weights(self.key)
        
        return VisionSTDPState(
            W=jnp.tile(W_init[None, :, :], (B, 1, 1)), # Shape: (B, Hidden, Polarized)
            e_pre=jnp.zeros((B, self.n_polarized), dtype=jnp.float32),
            e_post=jnp.zeros((B, self.n_hidden), dtype=jnp.float32),
            v_hidden=jnp.zeros((B, self.n_hidden), dtype=jnp.float32),
            v_th_adapt=jnp.zeros((B, self.n_hidden), dtype=jnp.float32),
            spike_trace=jnp.zeros((B, self.n_hidden), dtype=jnp.float32)
        )

    def __call__(self, state: VisionSTDPState, time_surface, tof_dist, learn=True):
        i_syn = jnp.einsum('bi,boi->bo', time_surface, state.W)
        i_ext = i_syn

        new_v_th_adapt = state.v_th_adapt * THETA_DECAY
        v_th_eff = V_TH_STDP + new_v_th_adapt 
        
        v_pre = lif_step(state.v_hidden, i_ext)
        spike_raw = jnp.maximum(v_pre - v_th_eff, 0.0)
        spike_out = localized_inhibition(spike_raw, pool_size=16, k_per_pool=1) 

        new_v_hidden = jnp.where(spike_out > 0.5, 0.0, v_pre)
        new_v_th_adapt = jnp.clip(new_v_th_adapt + THETA_INC * spike_out, 0.0, THETA_MAX)

        W_learned, e_pre_learned, e_post_learned = self.stdp_layer.apply_batch(state.W, state.e_pre, state.e_post, time_surface, spike_out)

        # learning gate (traced, so no Python control flow)
        W_final = jnp.where(learn, W_learned, state.W)
        e_pre_final = jnp.where(learn, e_pre_learned, TRACE_DECAY * state.e_pre + time_surface)
        e_post_final = jnp.where(learn, e_post_learned, TRACE_DECAY * state.e_post + spike_out)

        new_spike_trace = state.spike_trace * 0.95 + spike_out
        # normalise by the per-sample maximum so features stay in [0, 1]
        features = new_spike_trace / (jnp.max(new_spike_trace, axis=1, keepdims=True) + 1e-8)

        new_state = VisionSTDPState(W_final, e_pre_final, e_post_final, new_v_hidden, new_v_th_adapt, new_spike_trace)
        return new_state, spike_out, features
