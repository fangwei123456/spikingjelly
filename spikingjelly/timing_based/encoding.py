"""
Gaussian Receptive Field Population Coding
==========================================

| All neurons are arranged in a grid where each row corresponds to an input dimension
| and each column corresponds to a neuron in that dimension.
| When an input value is given, each neuron computes its response based on its Gaussian
| tuning curve.
| The response is then mapped to a spike time, where a higher response results in an
| earlier spike time, and a lower response results in a later spike time.
| If the spike time exceeds a predefined maximum spike time, the neuron becomes inactive
| (no spike, represented by -1).

Reference:
   | Sander M. Bohte, Joost N. Kok, Han La Poutré,
   | Error-backpropagation in temporally encoded networks of spiking neurons,
   | Neurocomputing,
   | Volume 48, Issues 1–4,
   | 2002,
   | Pages 17-37,
   | ISSN 0925-2312,
   | https://doi.org/10.1016/S0925-2312(01)00658-0.
   | (https://www.sciencedirect.com/science/article/pii/S0925231201006580)

Neuron Spatial Receptive Field (Lecture):
   | https://youtu.be/fCqt07IXUPI?si=jVT-QlmEgrbQZkB2
"""

from typing import Annotated

import einops
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _GaussianTuningConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    neurons_count: Annotated[
        int,
        Field(
            description="The number of neurons per input channel.",
            gt=2,
        ),
    ] = 5

    max_spike_time: Annotated[
        int,
        Field(
            description="""
            The maximum spike time for the neurons
            beyond which neurons become inactive.
            """,
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
        if not torch.all(self.input_min < self.input_max):
            raise ValueError("All elements of input_min must be less than input_max.")

        return self


class _GaussianTuning:
    def __init__(self, cfg: dict, device: torch.device) -> None:
        self.cfg = _GaussianTuningConfig(**cfg) if cfg else _GaussianTuningConfig()
        self.device = device
        self._build_receptive_fields()

    def encode(self, x: torch.Tensor, max_spike_time: int = 50) -> torch.Tensor:
        # ---- Input validation -------------------------------------------------------
        assert x.dim() == 3 and x.shape[1] == self.cfg.channels_count, (
            "Input tensor x must be 3-dimensional"
            "(batch_size, channels_count, samples_count)."
            f"Got shape: {x.shape}."
        )
        # Due to reshaping, we need to know the original shape for output formatting
        x_shape_orig = x.shape
        # ---- Input reshaping --------------------------------------------------------
        x = einops.rearrange(x, "b c s -> (b s) c")
        x = einops.repeat(x, "bs c -> bs c n", n=self.cfg.neurons_count)
        # ---- Neuron Responses -------------------------------------------------------
        neuron_responses = self._calculate_neurons_response(
            x,
            self.neuron_centers,
            self.neuron_variances,
        )
        # ---- Spike time mapping -----------------------------------------------------
        # higher response -> earlier spike time | lower response -> later spike time
        max_spike_time = max_spike_time or self.cfg.max_spike_time
        spike_times = (max_spike_time * (1 - neuron_responses)).round()
        spike_times[spike_times >= max_spike_time] = -1  # inactive neurons
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

    def _build_receptive_fields(self) -> None:
        """
        Build the Gaussian receptive fields for each neuron in each input dimension.
        Each neuron is defined by its center (mu) and variance (sigma^2).
        """
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
    def _calculate_neurons_centers(
        in_min: torch.Tensor,
        in_range: torch.Tensor,
        indices: torch.Tensor,
        neurons_count: int,
    ) -> torch.Tensor:
        """
        mu = in_min + ((2 * indices - 3) / (2 * (neurons_count - 2))) * in_range
        where:
            indices: neuron index (1 to m)
            neurons_count: number of neurons
            in_min: minimum value of the input data
            in_range: range of the input data (in_max - in_min)
        """
        nominator = 2 * indices - 3
        denominator = 2 * (neurons_count - 2)
        centers = in_min + (nominator / denominator) * in_range

        return centers

    @staticmethod
    def _calculate_neurons_variances(
        in_range: torch.Tensor,
        neurons_count: int,
        beta: float,
    ) -> torch.Tensor:
        """
        sigma^2 = ( (in_range / (beta * (neurons_count - 2)) ) )^2
        where:
            neurons_count: number of neurons
            in_range: range of the input data (in_max - in_min)
            beta: sharpness of the tuning curves
        """
        nominator = in_range
        denominator = beta * (neurons_count - 2)
        variances = (nominator / denominator).square()

        return variances

    @staticmethod
    def _calculate_neurons_response(
        x: torch.Tensor,
        centers: torch.Tensor,
        variances: torch.Tensor,
    ) -> torch.Tensor:
        """
        neuron_response = exp( -( (x - centers)^2 / (2 * variances) ) )
        where:
            x: input data
            centers: neuron centers (mu)
            variances: neuron variances (sigma^2)
        """
        nominator = (x - centers).square()
        denominator = 2 * variances
        neuron_responses = torch.exp(-(nominator / denominator))

        return neuron_responses


# NOTE: Facade class for backward compatibility
class GaussianTuning:
    def __init__(
        self,
        n: int,
        m: int,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        n : int
            The number of channels/dims in the input data.
        m : int
            The number of neurons per input channel.
        x_min : torch.Tensor
            1D tensor of shape (n,) representing the minimum value of the input data
            for each input dimension.
        x_max : torch.Tensor
            1D tensor of shape (n,) representing the maximum value of the input data
            for each input dimension.

        Raises
        ------
        ValueError
            If x_min and x_max do not have shape (n,)
            or if any element in x_min is not less than the corresponding element in x_max.

        Example
        -------
        .. code-block:: python

            >>> import torch
            >>> from spikingjelly.timing_based import encoding
            >>> x_min = torch.tensor([0.0])
            >>> x_max = torch.tensor([1.0])
            >>> encoder = encoding.GaussianTuning(n=1, m=4, x_min=x_min, x_max=x_max)
            >>> x = torch.tensor([[[0.1, 0.5, 0.9]]])
            >>> spikes = encoder.encode(x, max_spike_time=100)
            >>> print(spikes)
            tensor([[[[42., 10., 85., -1.],
                      [92., 25., 25., 92.],
                      [-1., 85., 10., 42.]]]])
            >>> print(spikes.shape)
            torch.Size([1, 1, 3, 4]) # (batch_size, channels_count, samples_count, neurons_count)

        An array/grid of Neuronal Receptive Fields::

            Each one is a Gaussian curve defined by its center (mu) and variance (sigma^2).
            The grid is made up of m neurons for each of the n input dimensions.

            ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐
            │ 1 │ │ 2 │ │ 3 │ │ 4 │ ... │ m │ <- m neurons (m=5) for Dimension 0
            └───┘ └───┘ └───┘ └───┘     └───┘
            ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐
            │ 1 │ │ 2 │ │ 3 │ │ 4 │ ... │ m │ <- m neurons (m=5) for Dimension 1
            └───┘ └───┘ └───┘ └───┘     └───┘
             ...   ...   ...   ...       ...
            ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐
            │ 1 │ │ 2 │ │ 3 │ │ 4 │ ... │ m │ <- m neurons (m=5) for Dimension n
            └───┘ └───┘ └───┘ └───┘     └───┘

        Each neuron computes the response to the input based on its Gaussian tuning curve::

                 0.1      0.1      0.1    0.1
                 0.5      0.5      0.5    0.5
                 0.9      0.9      0.9    0.9
                  │        │        │      │
                  ▼        ▼        ▼      ▼
                ┌───┐    ┌───┐    ┌───┐   ┌───┐
                │ 1 │    │ 2 │    │ 3 │   │ 4 │   <-- 4 neurons
                └───┘    └───┘    └───┘   └───┘
                  │        │        │       │
                  ▼        ▼        ▼       ▼
                 0.5762   0.9037   0.1494  0.0026
                 0.0796   0.7548   0.7548  0.0796 <-- responses (probability of firing)
                 0.0026   0.1494   0.9037  0.5762
                  │        │        │       │
                  ▼        ▼        ▼       ▼
                  42       10       85     -1
                  92       25       25      92    <-- spike times
                 -1        85       10      42

            lower response  -> later spike time
            higher response -> earlier spike time

            if the spike time >= max_spike_time, neuron becomes inactive (no spike, -1)

        Example
        -------
        .. code-block:: python

            >>> x_min = torch.tensor([0.0, 0.0, 0.0])
            >>> x_max = torch.tensor([1.0, 1.0, 1.0])
            >>> encoder = GaussianTuning(n=3, m=5, x_min=x_min, x_max=x_max)
            >>> x = torch.tensor([[[0.1, 0.5, 0.9], [0.2, 0.6, 0.8], [0.3, 0.7, 0.4]]])
            >>> spikes = encoder.encode(x, max_spike_time=100)
            >>> print(spikes)
            tensor([[[[51.,  4., 80., -1., -1.],
                      [99., 68.,  0., 68., 99.],
                      [-1., -1., 80.,  4., 51.]],

                     [[74.,  1., 60., 98., -1.],
                      [-1., 85., 10., 42., 96.],
                      [-1., 98., 60.,  1., 74.]],

                     [[89., 16., 33., 94., -1.],
                      [-1., 94., 33., 16., 89.],
                      [96., 42., 10., 85., -1.]]]])
            >>> print(spikes.shape)
            torch.Size([1, 3, 3, 5]) # (batch_size, channels_count, samples_count, neurons_count)
        """
        cfg = dict(
            neurons_count=m,
            channels_count=n,
            input_min=x_min,
            input_max=x_max,
        )
        self._encoder = _GaussianTuning(cfg, device=x_min.device)
        self._set_attributes()

    def encode(
        self,
        x: torch.Tensor,
        max_spike_time: int = 50,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels_count, samples_count).
        max_spike_time : int
            The maximum spike time for the neurons
            beyond which neurons become inactive.

        Returns
        -------
        torch.Tensor
            Encoded spike times of shape
            (batch_size, channels_count, samples_count, neurons_count).

        Raises
        ------
        AssertionError
            If the input tensor x does not have shape
            (batch_size, channels_count, samples_count).
        """
        encoded = self._encoder.encode(x, max_spike_time)

        return encoded

    def _set_attributes(self) -> None:
        self.m = self._encoder.cfg.neurons_count
        self.n = self._encoder.cfg.channels_count
        self.mu = self._encoder.neuron_centers
        self.sigma2 = self._encoder.neuron_variances
