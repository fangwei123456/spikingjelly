import numpy as np
from matplotlib import pyplot as plt
def plot_bar_and_text(x, y, width, label):
    plt.bar(x, y, width=width, label=label)
    for i in range(x.shape[0]):
        plt.text(x[i], y[i] + 0.2, str(round(y[i], 2)), ha='center')

plt.style.use(['science'])
fig = plt.figure(figsize=(8, 4), dpi=250)
csv_array = np.loadtxt('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time.csv', delimiter=',', skiprows=1)
T = csv_array[:, 0]
T_str = []
for i in range(T.shape[0]):
    T_str.append(str(int(T[i])))

t_cupy_f = csv_array[:, 2]
t_torch_f = csv_array[:, 1]
t_cupy_fb = csv_array[:, 4]
t_torch_fb = csv_array[:, 3]
x = np.arange(0, T.shape[0]) * 2
width = 0.3
plot_bar_and_text(x - width * 0.75, t_torch_f, width=width, label="backend='torch'")
plot_bar_and_text(x + width * 0.75, t_cupy_f, width=width, label="backend='cupy'")
plt.title('Execution time of Running Forward with $2^{20}$ Neurons')
plt.xlabel('simulation duration T (step)')
plt.ylabel('execution time (ms)')
plt.xticks(x, T_str)
plt.legend(frameon=True)
plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_f.svg')
plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_f.pdf')
plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_f.png')
plt.clf()
plot_bar_and_text(x - width * 0.75, t_torch_fb, width=width, label="backend='torch'")
plot_bar_and_text(x + width * 0.75, t_cupy_fb, width=width, label="backend='cupy'")
plt.title('Execution time of Running Forward and Backward with $2^{20}$ LIF Neurons')
plt.xlabel('simulation duration T (step)')
plt.ylabel('execution time (ms)')
plt.xticks(x, T_str)
plt.legend(frameon=True)
plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_fb.svg')
plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_fb.pdf')
plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_fb.png')