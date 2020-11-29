from matplotlib import pyplot as plt
import torch
from spikingjelly.clock_driven import neuron
from spikingjelly import visualizing
import numpy as np
import matplotlib

with torch.no_grad():
	# Requires SciencePlots package: https://github.com/garrettj403/SciencePlots 
	plt.style.use(['science'])

	if_node = neuron.IFNode(v_reset=None, monitor_state=True)
	T = 25
	x = torch.arange(0, 1.04, 0.04)
	for t in range(T):
		if_node(x)

	s_t_array = np.asarray(if_node.monitor['s']).T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
	v_t_array = np.asarray(if_node.monitor['v']).T  # v_t_array[i][j]表示神经元i在j时刻的电压值

	fig = plt.figure(dpi=125, tight_layout=True)
	gs = matplotlib.gridspec.GridSpec(2, 5)

	# plot_1d_spikes
	spikes_map = fig.add_subplot(gs[0, :4])
	firing_rate_map = fig.add_subplot(gs[0, 4])

	spikes_map.set_title('Spike Events')
	spikes_map.set_xlabel('t')
	spikes_map.set_ylabel('Neuron Index $i$')

	spikes_map.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
	spikes_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

	spikes_map.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	spikes_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

	spikes_map.set_xlim(0, s_t_array.shape[1] - 1)
	spikes_map.set_ylim(-0.5, s_t_array.shape[0] - 0.5)
	spikes_map.invert_yaxis()
	N = s_t_array.shape[0]
	T = s_t_array.shape[1]
	t = np.arange(0, T)
	t_spike = s_t_array * t
	mask = (s_t_array == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出

	colormap = plt.get_cmap('tab10')  # cmap的种类参见https://matplotlib.org/gallery/color/colormap_reference.html

	for i in range(N):
		spikes_map.eventplot(t_spike[i][mask[i]], lineoffsets=i, colors=colormap(i % 10))

	firing_rate = np.mean(s_t_array, axis=1, keepdims=True)
	max_rate = firing_rate.max()
	min_rate = firing_rate.min()
	firing_rate_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
	firing_rate_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	firing_rate_map.imshow(firing_rate, cmap='magma', aspect='auto')

	for i in range(firing_rate.shape[0]):
		firing_rate_map.text(0, i, f'{firing_rate[i][0]:.2f}', ha='center', va='center', color='w' if firing_rate[i][0] < 0.7 * max_rate or min_rate == max_rate else 'black')
	firing_rate_map.get_xaxis().set_visible(False)
	firing_rate_map.set_title('Firing Rate')

	# plot_2d_heatmap

	heatmap = fig.add_subplot(gs[1, :2])
	im = heatmap.imshow(v_t_array, aspect='auto', extent=[-0.5, T, v_t_array.shape[0] - 0.5, -0.5])
	heatmap.set_title('Membrane Potentials')
	heatmap.set_xlabel('t')
	heatmap.set_ylabel('Neuron Index $i$')
	heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
	heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
	heatmap.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	heatmap.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

	cbar = heatmap.figure.colorbar(im)
	cbar.ax.set_ylabel('Voltage Magnitude', rotation=90, va='top')
	cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

	# plot_2d_spiking_feature_map

	spikes = np.load('out_spikes_c.npy')
	C = spikes.shape[0]

	nrows = 8
	ncols = 16
	space = 1

	assert nrows * ncols == C, 'nrows * ncols != C'

	h = spikes.shape[1]
	w = spikes.shape[2]
	y = np.ones(shape=[(h + space) * nrows, (w + space) * ncols]) * spikes.max().item()
	index = 0
	for i in range(space // 2, y.shape[0], h + space):
		for j in range(space // 2, y.shape[1], w + space):
			y[i:i + h, j:j + w] = spikes[index]
			index += 1
	maps = fig.add_subplot(gs[1, 2:])
	maps.set_title('Spiking Feature Maps')
	maps.imshow(y, cmap='gray')

	maps.get_xaxis().set_visible(False)
	maps.get_yaxis().set_visible(False)

	plt.show()
