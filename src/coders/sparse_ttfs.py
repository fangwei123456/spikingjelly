"""Receptive Field Population Coding

Sparse Time-To-First-Spike (TTFS) Population Coding.

Reference:
    Sander M. Bohte, Joost N. Kok, Han La Poutré,
    Error-backpropagation in temporally encoded networks of spiking neurons,
    Neurocomputing,
    Volume 48, Issues 1–4,
    2002,
    Pages 17-37,
    ISSN 0925-2312,
    https://doi.org/10.1016/S0925-2312(01)00658-0.
    (https://www.sciencedirect.com/science/article/pii/S0925231201006580)

Neuron Spatial Receptive Field (Lecture):
    https://youtu.be/fCqt07IXUPI?si=jVT-QlmEgrbQZkB2
"""

from typing import Annotated

import einops
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SparseTTFSConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    neurons_count: Annotated[
        int,
        Field(
            description="The number of neurons per input dimension.",
            gt=2,
        ),
    ] = 5

    max_spike_time: Annotated[
        int,
        Field(
            description="The maximum spike time for the neurons. ",
            gt=0,
        ),
    ] = 100

    beta: Annotated[
        float,
        Field(
            description="The sharpness of the tuning curves.",
            gt=0.0,
        ),
    ] = 1.5

    channels_count: Annotated[
        int,
        Field(
            description="The number of channels/dims in the input data.",
            gt=0,
        ),
    ] = 1

    input_min: Annotated[
        torch.Tensor,
        Field(
            description="The minimum value of the input data.",
        ),
    ] = torch.tensor(0.0)

    input_max: Annotated[
        torch.Tensor,
        Field(
            description="The maximum value of the input data.",
        ),
    ] = torch.tensor(1.0)

    @model_validator(mode="after")
    def sanity_check(self):
        if not self.input_min.shape == self.input_max.shape == (self.channels_count,):
            raise ValueError(
                f"input_min and input_max should have shape ({self.channels_count},),"
                f" but got shapes {self.input_min.shape} and {self.input_max.shape}."
            )
        return self


class SparseTTFS:
    """
    An Array of Receptive Fields (A grid of Neurons):

    m: number of neurons per input dimension (e.g., m=5)
    n: number of input dimensions (e.g., n=3)

    ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐
    │ 1 │ │ 2 │ │ 3 │ │ 4 │ ... │ m │   <== m neurons (m=5) for Dimension 0
    └───┘ └───┘ └───┘ └───┘     └───┘
    ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐
    │ 1 │ │ 2 │ │ 3 │ │ 4 │ ... │ m │   <== m neurons (m=5) for Dimension 1
    └───┘ └───┘ └───┘ └───┘     └───┘
     ...   ...   ...   ...       ...
    ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐
    │ 1 │ │ 2 │ │ 3 │ │ 4 │ ... │ m │   <== m neurons (m=5) for Dimension n
    └───┘ └───┘ └───┘ └───┘     └───┘

    Each neuron is a Gaussian curve.


    Example
    -------

    # input shape = (1, 1, 3) (batch_size, channels_count, samples_count)
    x = torch.tensor([[[0.1, 0.5, 0.9]]])
    num_neurons = 4

    Each neuron will compute the response to each input value.

    0.1, 0.5, 0.9   0.1, 0.5, 0.9   0.1, 0.5, 0.9   0.1, 0.5, 0.9
          │               │               │               │
          ▼               ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
    │     1     │   │     2     │   │     3     │   │     4     │  <-- 4 neurons
    └───────────┘   └───────────┘   └───────────┘   └───────────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
      response        response        response        response <-- Probability of firing
      for each        for each        for each        for each
      input value     input value     input value     input value
          │               │               │               │
          ▼               ▼               ▼               ▼
     converting      converting      converting      converting
     to spike time   to spike time   to spike time   to spike time

    lower response -> later spike time
    higher response -> earlier spike time
    if the spike time >= max_spike_time, neuron becomes inactive (no spike, -1)
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = SparseTTFSConfig(**cfg) if cfg else SparseTTFSConfig()
        self.device = torch.device(device)
        self._build_receptive_fields()

    def encode(self, x):
        # ---- Input validation -------------------------------------------------------
        assert x.dim() == 3, (
            "Input tensor x must be 3-dimensional"
            "(batch_size, channels_count, samples_count)."
        )
        assert x.shape[1] == self.cfg.channels_count, (
            "Input tensor x must have"
            f"{self.cfg.channels_count} channels, but got {x.shape[1]}."
        )

        x_shape_orig = x.shape
        x = einops.rearrange(
            x,
            "b c s -> b s c",
        ).contiguous()  # (batch_size, samples_count, channels_count)
        x = einops.rearrange(
            x,
            "b s c -> (b s) c",
        )  # (batch_size * samples_count, channels_count)
        x = einops.repeat(
            x,
            "bs c -> bs c n",
            n=self.cfg.neurons_count,
        )  # (batch_size * samples_count, channels_count, neurons_count)
        # ---- Neuron Responses -------------------------------------------------------
        neuron_responses = self._calculate_neurons_response(
            x,
            self.neuron_centers,
            self.neuron_variances,
        )
        # ---- Spike time mapping -----------------------------------------------------
        # higher response -> earlier spike time
        # lower response -> later spike time
        spike_times = (self.cfg.max_spike_time * (1 - neuron_responses)).round()
        spike_times[spike_times >= self.cfg.max_spike_time] = -1  # inactive neurons
        # ---- Output formatting ------------------------------------------------------
        # Reshape back to (batch_size, samples_count, channels_count, neurons_count)
        batch_size = x_shape_orig[0]
        samples_count = x_shape_orig[2]
        out_spikes = einops.rearrange(
            spike_times,
            "(b s) c n -> b c s n",
            b=batch_size,
            s=samples_count,
        )

        return out_spikes

    def _build_receptive_fields(self):
        # A row of neuron indices [1, 2, ..., m] for each input dimension
        neuron_indices = (
            torch.arange(start=1, end=self.cfg.neurons_count + 1)  # A row of neurons
            .unsqueeze(0)  # add extra dimension to handle other input dimensions
            .repeat(self.cfg.channels_count, 1)  # repeat for each dimension
            .float()
            .to(self.device)
        )
        # Repeat input_min and input_max for each neurons row
        _input_min = self.cfg.input_min.unsqueeze(-1).repeat(1, self.cfg.neurons_count)
        _input_max = self.cfg.input_max.unsqueeze(-1).repeat(1, self.cfg.neurons_count)
        _input_range = _input_max - _input_min
        # Calculate the centers and variances for each neuron in each dimension
        self.neuron_centers = self._calculate_neurons_centers(
            _input_min,
            _input_range,
            neuron_indices,
            self.cfg.neurons_count,
        )
        self.neuron_variances = self._calculate_neurons_variances(
            _input_range,
            self.cfg.neurons_count,
            self.cfg.beta,
        )

    @staticmethod
    def _calculate_neurons_centers(in_min, in_range, indices, neurons_count):
        centers = in_min + ((2 * indices - 3) / 2) * (in_range / (neurons_count - 2))

        return centers

    @staticmethod
    def _calculate_neurons_variances(in_range, neurons_count, beta):
        variances = (1 / ((beta * in_range) * (neurons_count - 2))) ** 2

        return variances

    @staticmethod
    def _calculate_neurons_response(x, centers, variances):
        neuron_responses = torch.exp(-((x - centers) ** 2) / (2 * variances))

        return neuron_responses
