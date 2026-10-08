#!/usr/bin/env python3
"""Dual-stream vision front end: the frozen CSNN stream and the plastic STDP stream, both driven by the
event time surface."""

import os
import jax
import jax.numpy as jnp
import flax.serialization
from typing import NamedTuple

from snn_vision_csnn import CSNN_Base 
from snn_vision_stdp import VisionSTDP, VisionSTDPState
from src.sparse_forest import N_PIXELS 

# Front-end state (nested tuple, so the whole front end is a pure function of its state)
class FusionState(NamedTuple):
    time_surface: jnp.ndarray      
    csnn_trace: jnp.ndarray        
    stdp_trace: jnp.ndarray        
    stdp_state: VisionSTDPState         

class DualStreamVisionCortex:
    def __init__(self, key, n_pixels=N_PIXELS, n_csnn_out=256, n_stdp_out=256):
        self.n_pixels = n_pixels
        self.n_csnn_out = n_csnn_out
        self.n_stdp_out = n_stdp_out
        
        k_stdp, k_csnn = jax.random.split(key)
        
        self.csnn = CSNN_Base() 
        self.stdp = VisionSTDP(key=k_stdp, n_input=n_pixels, n_hidden=n_stdp_out)
        
        weights_path = os.path.join(os.path.dirname(__file__), "frozen_csnn_weights.msgpack")
        if os.path.exists(weights_path):
            with open(weights_path, "rb") as f:
                self.frozen_params = flax.serialization.msgpack_restore(f.read())
            print("Dual-Stream Vision: loaded frozen CSNN weights.")
        else:
            dummy_ev = jnp.zeros((1, n_pixels * 2))
            dummy_tof = jnp.zeros((1, 1))           
            self.frozen_params = self.csnn.init(k_csnn, dummy_ev, dummy_tof)
            print(f"WARNING: {weights_path} not found. Using randomly initialised weights.")

    def init_state(self, B) -> FusionState:
        return FusionState(
            time_surface=jnp.zeros((B, self.n_pixels * 2)), 
            csnn_trace=jnp.zeros((B, self.n_csnn_out)),
            stdp_trace=jnp.zeros((B, self.n_stdp_out)),
            stdp_state=self.stdp.init_state(B)
        )

    def __call__(self, state: FusionState, ev_frame, tof_dist, learn=True):
        on_spike = jnp.clip(ev_frame, 0.0, 1.0)
        off_spike = jnp.clip(-ev_frame, 0.0, 1.0)
        ev_pol = jnp.concatenate([on_spike, off_spike], axis=1)
        
        new_time_surface = state.time_surface * 0.8 + ev_pol
        
        # Both streams read the same time surface; ToF is scaled to the CSNN's training range
        csnn_raw_out = self.csnn.apply(self.frozen_params, new_time_surface, tof_dist / 8.0)
        new_stdp_state, stdp_spikes, stdp_features = self.stdp(state.stdp_state, new_time_surface, tof_dist, learn=learn)
        
        BETA_OUT = 0.95
        new_csnn_trace = BETA_OUT * state.csnn_trace + (1.0 - BETA_OUT) * csnn_raw_out
        # Integrate the normalised STDP spike trace (continuous from the first frame) rather than raw spikes
        new_stdp_trace = BETA_OUT * state.stdp_trace + (1.0 - BETA_OUT) * stdp_features
        
        # The CSNN output (a dense layer) has no spatial topography, so it is not blurred
        csnn_clean = jnp.maximum(0.0, new_csnn_trace)
        
        # The STDP features are retinotopic: blur them slightly across neighbouring neurons
        new_stdp_trace_clean = jnp.maximum(0.0, new_stdp_trace)
        stdp_left  = jnp.pad(new_stdp_trace_clean[:, :-1], ((0, 0), (1, 0)), mode='constant')
        stdp_right = jnp.pad(new_stdp_trace_clean[:, 1:],  ((0, 0), (0, 1)), mode='constant')
        stdp_blurred = (new_stdp_trace_clean + 0.05 * stdp_left + 0.05 * stdp_right) / 1.10
        
        # Normalize both independently
        norm_csnn = csnn_clean / (jnp.linalg.norm(csnn_clean, axis=-1, keepdims=True) + 1e-8)
        norm_stdp = stdp_blurred / (jnp.linalg.norm(stdp_blurred, axis=-1, keepdims=True) + 1e-8)

        new_state = FusionState(new_time_surface, new_csnn_trace, new_stdp_trace, new_stdp_state)

        return new_state, (norm_csnn, norm_stdp)