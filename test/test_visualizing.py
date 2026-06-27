import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from spikingjelly import visualizing


class TestPlot2dHeatmap:
    def test_numpy_input(self):
        arr = np.random.rand(20, 10)
        fig, ax = visualizing.plot_2d_heatmap(arr, title="test", xlabel="x", ylabel="y")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_torch_input(self):
        t = torch.rand(20, 10)
        fig, ax = visualizing.plot_2d_heatmap(t, title="test", xlabel="x", ylabel="y")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_x_max(self):
        arr = np.random.rand(20, 10)
        fig, ax = visualizing.plot_2d_heatmap(
            arr, title="t", xlabel="x", ylabel="y", x_max=50
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_no_colorbar(self):
        arr = np.random.rand(20, 10)
        fig, ax = visualizing.plot_2d_heatmap(
            arr, title="t", xlabel="x", ylabel="y", plot_colorbar=False
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_raises_on_1d(self):
        with pytest.raises(ValueError, match="2D"):
            visualizing.plot_2d_heatmap(np.zeros(10), title="t", xlabel="x", ylabel="y")


class TestPlot2dBarIn3d:
    def test_numpy_input(self):
        arr = np.random.rand(5, 10)
        fig, ax = visualizing.plot_2d_bar_in_3d(
            arr, title="t", xlabel="x", ylabel="y", zlabel="z"
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_torch_input(self):
        t = torch.rand(5, 10)
        fig, ax = visualizing.plot_2d_bar_in_3d(
            t, title="t", xlabel="x", ylabel="y", zlabel="z"
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_raises_on_1d(self):
        with pytest.raises(ValueError, match="2D"):
            visualizing.plot_2d_bar_in_3d(
                np.zeros(10), title="t", xlabel="x", ylabel="y", zlabel="z"
            )


class TestPlot1dSpikes:
    def test_numpy_input(self):
        spikes = (np.random.rand(50, 10) > 0.8).astype(float)
        fig, ax = visualizing.plot_1d_spikes(spikes, title="t", xlabel="x", ylabel="y")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_torch_input(self):
        spikes = (torch.rand(50, 10) > 0.8).float()
        fig, ax = visualizing.plot_1d_spikes(spikes, title="t", xlabel="x", ylabel="y")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_no_firing_rate(self):
        spikes = (np.random.rand(50, 10) > 0.8).astype(float)
        fig, ax = visualizing.plot_1d_spikes(
            spikes, title="t", xlabel="x", ylabel="y", plot_firing_rate=False
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_raises_on_1d(self):
        with pytest.raises(ValueError, match="2D"):
            visualizing.plot_1d_spikes(np.zeros(10), title="t", xlabel="x", ylabel="y")


class TestPlot2dFeatureMap:
    def test_numpy_input(self):
        x3d = np.random.rand(12, 8, 8)
        fig, ax = visualizing.plot_2d_feature_map(
            x3d, nrows=3, ncols=4, space=2, title="t"
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_torch_input(self):
        x3d = torch.rand(12, 8, 8)
        fig, ax = visualizing.plot_2d_feature_map(
            x3d, nrows=3, ncols=4, space=2, title="t"
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_raises_on_2d(self):
        with pytest.raises(ValueError, match="3D"):
            visualizing.plot_2d_feature_map(
                np.zeros((8, 8)), nrows=2, ncols=2, space=1, title="t"
            )

    def test_raises_on_mismatch(self):
        with pytest.raises(ValueError, match="nrows"):
            visualizing.plot_2d_feature_map(
                np.random.rand(12, 8, 8), nrows=2, ncols=2, space=1, title="t"
            )


class TestPlotOneNeuronVS:
    def test_numpy_input(self):
        v = np.sin(np.linspace(0, 10, 100))
        s = (np.random.rand(100) > 0.9).astype(float)
        fig, ax_v, ax_s = visualizing.plot_one_neuron_v_s(v, s)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax_v, matplotlib.axes.Axes)
        assert isinstance(ax_s, matplotlib.axes.Axes)
        plt.close("all")

    def test_torch_input(self):
        v = torch.sin(torch.linspace(0, 10, 100))
        s = (torch.rand(100) > 0.9).float()
        fig, ax_v, ax_s = visualizing.plot_one_neuron_v_s(v, s)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax_v, matplotlib.axes.Axes)
        assert isinstance(ax_s, matplotlib.axes.Axes)
        plt.close("all")

    def test_v_reset_none(self):
        v = np.sin(np.linspace(0, 10, 100))
        s = (np.random.rand(100) > 0.9).astype(float)
        fig, ax_v, ax_s = visualizing.plot_one_neuron_v_s(v, s, v_reset=None)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_raises_on_2d_v(self):
        with pytest.raises(ValueError, match="1D"):
            visualizing.plot_one_neuron_v_s(np.zeros((10, 5)), np.zeros(10))

    def test_raises_on_2d_s(self):
        with pytest.raises(ValueError, match="1D"):
            visualizing.plot_one_neuron_v_s(np.zeros(10), np.zeros((10, 5)))
