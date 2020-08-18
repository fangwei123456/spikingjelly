from matplotlib import pyplot as plt
import torch
from spikingflow.clock_driven import neuron
from spikingflow import visualizing
import numpy as np
import matplotlib

# Requires SciencePlots package: https://github.com/garrettj403/SciencePlots 
plt.style.use(['science'])

neuron_num = 32
T = 50
lif_node = neuron.LIFNode(monitor_state=True)
w = torch.rand([neuron_num]) * 50
for t in range(T):
	lif_node(w * torch.rand(size=[neuron_num]))
s_t_array = np.asarray(lif_node.monitor['s']).T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
v_t_array = np.asarray(lif_node.monitor['v']).T  # v_t_array[i][j]表示神经元i在j时刻的电压值

fig = plt.figure(dpi=125, tight_layout=True)
gs = matplotlib.gridspec.GridSpec(5, 5)

# plot_1d_spikes
spikes_map = fig.add_subplot(gs[:3, :4])
spiking_rate_map = fig.add_subplot(gs[:3, 4])

spikes_map.set_title('Spikes of Neurons')
spikes_map.set_xlabel('Simulating Step')
spikes_map.set_ylabel('Neuron Index')

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

spiking_rate = np.mean(s_t_array, axis=1, keepdims=True)
max_rate = spiking_rate.max()
min_rate = spiking_rate.min()
spiking_rate_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
spiking_rate_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
spiking_rate_map.imshow(spiking_rate, cmap='magma', aspect='auto')

for i in range(spiking_rate.shape[0]):
	spiking_rate_map.text(0, i, f'{spiking_rate[i][0]:.2f}', ha='center', va='center', color='w' if spiking_rate[i][0] < 0.7 * max_rate or min_rate == max_rate else 'black')
spiking_rate_map.get_xaxis().set_visible(False)
spiking_rate_map.set_title('Firing Rate')

# plot_2d_heatmap

heatmap = fig.add_subplot(gs[3:, :3])
im = heatmap.imshow(v_t_array, aspect='auto', extent=[-0.5, T, v_t_array.shape[0] - 0.5, -0.5])
heatmap.set_title('Membrane Potentials')
heatmap.set_xlabel('Simulating Step')
heatmap.set_ylabel('Neuron Index')
heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
heatmap.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
heatmap.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

cbar = heatmap.figure.colorbar(im)
cbar.ax.set_ylabel('Voltage Magnitude', rotation=90, va='top')
cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

# plot_2d_spiking_feature_map

C = 48
W = 8
H = 8
spikes = (np.random.rand(C, W, H) > 0.8).astype(float)
C = spikes.shape[0]

h = spikes.shape[1]
w = spikes.shape[2]
y = np.ones(shape=[(h + 2) * 6, (w + 2) * 8]) * spikes.max().item()
index = 0
for i in range(1, y.shape[0], h + 2):
	for j in range(1, y.shape[1], w + 2):
		y[i:i + h, j:j + w] = spikes[index]
		index += 1
maps = fig.add_subplot(gs[3:, 3:])
maps.set_title('Spiking Feature Maps')
maps.imshow(y, cmap='gray')

maps.get_xaxis().set_visible(False)
maps.get_yaxis().set_visible(False)

plt.show()
