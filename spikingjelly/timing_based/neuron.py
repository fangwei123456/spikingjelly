"""
Tempotron Neuron Model
======================

| Tempotron is a Leaky Integrate-and-Fire (LIF) Neuron Model,
| that accepts spikes from sensory neurons spikes
| and learns to classify spatiotemporal patterns of those spikes.

Reference:
    | Gütig R, Sompolinsky H.
    | The tempotron: a neuron that learns spike timing-based decisions.
    | Nat Neurosci.
    | 2006 Mar;9(3):420-8.
    | DOI: 10.1038/nn1643.
    | Epub 2006 Feb 12.
    | PMID: 16474393.
"""

from typing import Annotated

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, Field


class _TempotronConfig(BaseModel):
    in_neurons_count: Annotated[
        int,
        Field(
            description="Number of input neurons",
            ge=1,
        ),
    ] = 1
    out_neurons_count: Annotated[
        int,
        Field(
            description="Number of output neurons",
            ge=1,
        ),
    ] = 1
    time_window: Annotated[
        int,
        Field(
            description="Temporal window to consider for spike patterns",
            ge=1,
        ),
    ] = 100
    tau: Annotated[
        float,
        Field(
            description="Decay time constant",
            gt=0,
        ),
    ] = 15.0
    tau_s: Annotated[
        float,
        Field(
            description="Synaptic current time constant",
            gt=0,
        ),
    ] = (
        15.0 / 4
    )
    threshold_voltage: Annotated[
        float,
        Field(
            description="Membrane threshold voltage",
            gt=0,
        ),
    ] = 1.0


class _Tempotron(nn.Module):
    def __init__(self, cfg=None, device=torch.device("cpu")):
        self.cfg = _TempotronConfig(**cfg) if cfg else _TempotronConfig()
        self.device = device
        super().__init__()
        self._init()

    def _init(self):
        # time at which PSP kernel reaches its maximum value
        self.max_v_time = self.max_voltage_time(self.cfg.tau, self.cfg.tau_s)
        self.v_norm_factor = self.voltage_normalization_factor(
            self.max_v_time,
            self.cfg.tau,
            self.cfg.tau_s,
            self.cfg.threshold_voltage,
        )
        # to sum the contributions from all input neurons to get the output voltage
        self.summation_layer = nn.Linear(
            self.cfg.in_neurons_count,
            self.cfg.out_neurons_count,
            bias=False,
        )

    def forward(self, in_spikes_timings, out_voltage_type):
        batch_size = in_spikes_timings.shape[0]

        # to find the voltage contribution from each input spike at each time step t
        # we need to repeate the time steps for each sensory neuron and batch
        temporal_window = (
            torch.arange(0, self.cfg.time_window)  # time steps
            .view(1, 1, -1)  # extra dims for batch and in_neurons
            .repeat(
                batch_size,
                self.cfg.in_neurons_count,
                1,
            )
            .to(self.device)
        )
        # input spikes timings need to be repeated for each time step t
        in_spikes_timings = in_spikes_timings.unsqueeze(-1).repeat(
            1, 1, self.cfg.time_window
        )

        # voltage contribution from each input spike at each time step t
        in_voltage = self.input_voltage_contribution(
            temporal_window - in_spikes_timings,
            in_spikes_timings,
        )  # batch_size, in_neurons_count, time_window

        # total output voltage at each time step t
        out_voltage = self.summation_layer(rearrange(in_voltage, "b n t -> b t n"))
        out_voltage = rearrange(out_voltage, "b t n -> b n t")

        return self.interpreted_out_voltage(out_voltage, out_voltage_type)

    def mse_loss(self, v_max, label):
        wrong_mask = (
            (v_max >= self.cfg.threshold_voltage).float()
            != F.one_hot(label, self.cfg.out_neurons_count)
        ).float()
        squared_error = torch.pow((v_max - self.cfg.threshold_voltage) * wrong_mask, 2)
        mse = torch.sum(squared_error) / label.shape[0]

        return mse

    def input_voltage_contribution(self, delta, in_spikes_timings):
        input_voltage = (
            self.v_norm_factor
            * self.psp_kernel(
                delta,
                self.cfg.tau,
                self.cfg.tau_s,
            )
            * self.heaviside(in_spikes_timings)
        )

        return input_voltage

    def interpreted_out_voltage(self, out_v, out_voltage_type):
        match out_voltage_type:
            case "v":
                return out_v

            case "v_max":
                return F.max_pool1d(out_v, kernel_size=self.cfg.time_window).squeeze()

            case "spikes":
                batch_size = out_v.shape[0]
                temporal_window = (
                    torch.arange(0, self.cfg.time_window)  # time steps
                    .view(1, 1, -1)  # extra dims for batch and in_neurons
                    .repeat(
                        batch_size,
                        self.cfg.out_neurons_count,
                        1,
                    )
                    .to(self.device)
                )
                max_index = out_v.argmax(dim=2)
                max_index_soft = (
                    F.softmax(out_v * self.cfg.time_window, dim=2) * temporal_window
                ).sum(dim=2)
                v_max = F.max_pool1d(out_v, kernel_size=self.cfg.time_window).squeeze()
                mask = (v_max >= self.cfg.threshold_voltage).float() * 2 - 1
                max_index = max_index * mask
                max_index_soft = max_index_soft * mask
                return max_index_soft + (max_index - max_index_soft).detach()

            case _:
                raise ValueError(
                    f"Invalid out_voltage_type: {out_voltage_type}."
                    "Must be 'v', 'v_max', or 'spikes'"
                )

    def voltage_normalization_factor(self, max_v_time, tau, tau_s, threshold):
        """
        The normalization factor
        to make the maximum of PSP kernel equal to v_threshold at t_max
        and is calculated by setting K(t_max) = v_threshold

        K(t_max) = exp(-t_max/tau) - exp(-t_max/tau_s) = v_threshold
        V0 = v_threshold / K(t_max)
        """
        v_t_max = self.post_synaptic_potential(max_v_time, tau, tau_s)
        v0 = threshold / v_t_max

        return v0

    def psp_kernel(self, delta: torch.Tensor, tau, tau_s):
        """
        Post-Synaptic Potential (PSP) kernel

        K(Δt) = exp(-Δt/tau) - exp(-Δt/tau_s) , Δt = t - t_i

        Heaviside function H(Δt) is used to discard negative time differences::
            as they shouldn't contribute to the post-synaptic potential
            (Tempotron only responds to spikes that have already occurred)
        """
        # Heaviside discards negative time differences
        K = self.heaviside(delta) * self.post_synaptic_potential(delta, tau, tau_s)

        return K

    @staticmethod
    def post_synaptic_potential(delta: torch.Tensor, tau, tau_s):
        """
        Post-Synaptic Potential (PSP)
        """
        psp = torch.exp(-delta / tau) - torch.exp(-delta / tau_s)

        return psp

    @staticmethod
    def max_voltage_time(tau, tau_s):
        """
        K(t) = exp(-t/tau) - exp(-t/tau_s)  PSP kernel

        To find t_max where K(t) is maximum, set the derivative to zero:
        dK/dt = -1/tau * exp(-t/tau) + 1/tau_s * exp(-t/tau_s) = 0

                 tau * tau_s * log(tau / tau_s)
        t_max = -------------------------------- (derivative of PSP kernel)
                       (tau - tau_s)
        """
        t_max = (tau * tau_s * torch.log(torch.tensor(tau / tau_s))) / (tau - tau_s)

        return t_max

    @staticmethod
    def heaviside(x):
        """
        H(x) = 1, x >= 0
        H(x) = 0, x < 0

                    ▲
                    ├=========
              x<0   │ x>=0
           ◄─=======┼────────►
                    │
                    │
                    ▼
        """
        return (x >= 0).float()


# NOTE: Facade class for backward compatibility
class Tempotron(nn.Module):
    """
    Neuronal Simulation::

                         ┌─────────────────────┐
        Time:            │0 1 2 3 4 5 6 7 8 T-1│10 11 12 13 14 15 16 17 18 19 20 ......
        Sensory Neuron 1:│------------|--------│---------------------------------------
        Sensory Neuron 2:│----|----------------│---------------------------------------
                         └─────────────────────┘
                               Time Window

    Tempotron Neuron accepts timing of spikes
    from sensory neurons within a defined time window
    and learns to classify different spatiotemporal patterns of those spikes.

    Tempotron doesn't consider the rate of incoming spikes,
    it consider the precise timing and spatial arrangement of incoming spikes.

    Something like this::

        -|--|------|--|----  vs  -|-----|-------|-|- (single neuron)

        both have the same number of spikes, but different timing patterns.
        Spike Patterns could be across multiple neurons too
    """

    def __init__(
        self,
        in_features,
        out_features,
        T,
        tau=15.0,
        tau_s=15.0 / 4,
        v_threshold=1.0,
    ):
        """
        Parameters
        ----------
        in_features : int
            Number of input neurons.
        out_features : int
            Number of output neurons.
        T : int
            Temporal window to consider for spike patterns.
        tau : float, optional
            Decay time constant, by default 15.0
        tau_s : float, optional
            Synaptic current time constant, by default 15.0 / 4
        v_threshold : float, optional
            Membrane threshold voltage, by default 1.0
        """
        super().__init__()
        cfg = {
            "in_neurons_count": in_features,
            "out_neurons_count": out_features,
            "time_window": T,
            "tau": tau,
            "tau_s": tau_s,
            "threshold_voltage": v_threshold,
        }
        self.model = _Tempotron(cfg=cfg)

    def forward(self, in_spikes, ret_type):
        """
        Parameters
        ----------
        in_spikes_timings : torch.Tensor
            Shape: (batch_size, in_neurons_count)
            The spike timings from sensory neurons.
        out_voltage_type : str
            The type of output voltage to return ['v', 'v_max', 'spikes'].

        Returns
        -------
        torch.Tensor
            The output voltage based on the specified type.

        Raises
        ------
        ValueError
            If an invalid out_voltage_type is provided.
            Should be one of 'v', 'v_max', or 'spikes'.
        """
        return self.model(in_spikes, ret_type)

    def mse_loss(self, v_max, label):
        """
        Mean Squared Error Loss for Tempotron Neuron

        wrong_mask: Identifies neurons that misclassified the input.
        A neuron is considered to have misclassified if:
            - It fired (v_max >= threshold) when it shouldn't have (not the correct class).
            - It didn't fire (v_max < threshold) when it should have (the correct class).

        loss: Computes the mean squared error of the voltage difference
        for the misclassified neurons, averaged over the batch size.

        Parameters
        ----------
        v_max : torch.Tensor
            Shape: (batch_size, out_neurons_count)
            The maximum voltage output from the Tempotron neuron.
        label : torch.Tensor
            Shape: (batch_size,)
            The true class labels for the input data.

        Returns
        -------
        torch.Tensor
        """
        return self.model.mse_loss(v_max, label)
