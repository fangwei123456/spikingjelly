"""Torch reference for CLIF's two differentiable transitions."""

import torch


def _charge(x, v, m, tau):
    h = v - v / tau + x
    return h, m * torch.sigmoid(h / tau)


def _reset(h, decayed_m, spike, threshold):
    m = decayed_m + spike
    return h - spike * (threshold + torch.sigmoid(m)), m


def _forward_impl(x, v, m, tau, threshold, spike_function):
    h, decayed_m = _charge(x, v, m, tau)
    spike = spike_function(h - threshold)
    v, m = _reset(h, decayed_m, spike, threshold)
    return spike, v, m
