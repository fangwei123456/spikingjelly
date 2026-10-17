"""Torch reference for CLIF's differentiable transitions."""

import torch


def _reset(h, spike, m, threshold):
    return h - spike * (threshold + torch.sigmoid(m))


def _forward_impl(x, v, m, tau, threshold, spike_function):
    h = v - (v - 0.0) / tau + x
    decayed_m = m * torch.sigmoid(h / tau)
    spike = spike_function(h - threshold)
    m = decayed_m + spike
    return spike, _reset(h, spike, m, threshold), m
