import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.clock_driven import neuron, encoding, functional
from spikingjelly import visualizing

T=100

plt.style.use(['science', 'muted'])

# 正确率
plt.figure(dpi=200)
ax1 = plt.subplot(211)
plt.tight_layout()
train_accs = np.load('train_accs.npy')
plt.plot(train_accs)
plt.title('Train Acc')
plt.xlabel('Iteration')
plt.ylabel('Acc')
max_x = train_accs.argmax()
max_y = train_accs[max_x]
plt.scatter(max_x, max_y, marker='*', color='red')
plt.annotate(f'({max_x},{max_y:.3})', fontsize=6, xycoords='data', xy=(max_x, max_y), textcoords="offset points", xytext=(10, 10),
		arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3', color="0.5", shrinkA=5, shrinkB=5))
plt.tight_layout()

ax2 = plt.subplot(212)
plt.tight_layout()
test_accs = np.load('test_accs.npy')
plt.plot(test_accs)
plt.title('Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
max_x = test_accs.argmax()
max_y = test_accs[max_x]
plt.scatter(max_x, max_y, marker='*', color='red')
plt.annotate(f'({max_x},{max_y:.3})', fontsize=6, xycoords='data', xy=(max_x, max_y), textcoords="offset points", xytext=(10, 10),
		arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3', color="0.5", shrinkA=5, shrinkB=5))
plt.tight_layout()
plt.subplots_adjust(wspace =0, hspace =.5)

plt.show()

# 输出层脉冲
v_t_array = np.load('v_t_array.npy')
visualizing.plot_2d_heatmap(array=v_t_array, title='Membrane Potentials', xlabel='Simulating Step',
							ylabel='Neuron Index', int_x_ticks=True, int_y_ticks=True,
							plot_colorbar=True, colorbar_y_label='Voltage Magnitude', x_max=T, dpi=200)
plt.show()

s_t_array = np.load('s_t_array.npy')
visualizing.plot_1d_spikes(spikes=s_t_array, title='Spikes of Neurons', xlabel='Simulating Step',
							ylabel='Neuron Index', int_x_ticks=True, int_y_ticks=True,
							plot_firing_rate=True, firing_rate_map_title='Firing Rate', dpi=200)
plt.show()