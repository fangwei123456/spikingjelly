"""Torch reference for CLIF's differentiable transitions."""

import torch


def _charge(x, v, tau):
    return v - v / tau + x


def _decay(h, m, tau):
    return m * torch.sigmoid(h / tau)


def _reset(h, spike, m, threshold):
    return h - spike * (threshold + torch.sigmoid(m))


def _forward_impl(x, v, m, tau, threshold, spike_function):
    h = _charge(x, v, tau)
    decayed_m = _decay(h, m, tau)
    spike = spike_function(h - threshold)
    m = decayed_m + spike
    return spike, _reset(h, spike, m, threshold), m
