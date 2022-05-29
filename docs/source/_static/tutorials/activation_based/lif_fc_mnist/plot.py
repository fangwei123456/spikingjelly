import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spikingjelly import visualizing

T=100

plt.style.use(['science', 'high-contrast'])

# Train Acc.
train_acc_df = pd.read_csv('train_acc.csv')
train_acc_x = np.array(train_acc_df['Step'])
train_acc_y = np.array(train_acc_df['Value'])

plt.figure(figsize=(8, 4), dpi=200)
ax1 = plt.subplot(211)
plt.tight_layout()
plt.plot(train_acc_x, train_acc_y)
plt.title('Train Acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
max_x = train_acc_y.argmax()
max_y = train_acc_y[max_x]
plt.scatter(max_x, max_y, marker='*', color='red')
plt.annotate(f'({max_x},{max_y:.3})', fontsize=6, xycoords='data', xy=(max_x, max_y), textcoords="offset points", xytext=(10, 10),
		arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3', color="0.5", shrinkA=5, shrinkB=5))
plt.tight_layout()

# Test Acc.
test_acc_x = []
test_acc_y = []

test_acc_df = pd.read_csv('test_acc.csv')
test_acc_x = np.array(test_acc_df['Step'])
test_acc_y = np.array(test_acc_df['Value'])

ax2 = plt.subplot(212)
plt.tight_layout()
plt.plot(test_acc_x, test_acc_y)
plt.title('Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
max_x = test_acc_y.argmax()
max_y = test_acc_y[max_x]
plt.scatter(max_x, max_y, marker='*', color='red')
plt.annotate(f'({max_x},{max_y:.3})', fontsize=6, xycoords='data', xy=(max_x, max_y), textcoords="offset points", xytext=(10, 10),
		arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3', color="0.5", shrinkA=5, shrinkB=5))
plt.tight_layout()
plt.subplots_adjust(wspace =0, hspace =.5)

plt.savefig('acc.pdf')

# 输出层脉冲
v_t_array = np.load('v_t_array.npy')
visualizing.plot_2d_heatmap(array=v_t_array, title='Membrane Potentials', xlabel='Simulating Step',
							ylabel='Neuron Index', int_x_ticks=True, int_y_ticks=True,
							plot_colorbar=True, colorbar_y_label='Voltage Magnitude', x_max=T, figsize=(8, 6), dpi=200)
plt.savefig('2d_heatmap.pdf')

s_t_array = np.load('s_t_array.npy')
visualizing.plot_1d_spikes(spikes=s_t_array, title='Spikes of Neurons', xlabel='Simulating Step',
							ylabel='Neuron Index', int_x_ticks=True, int_y_ticks=True,
							plot_firing_rate=True, firing_rate_map_title='Firing Rate', figsize=(8, 6), dpi=200)
plt.savefig('1d_spikes.pdf')