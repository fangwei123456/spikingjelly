import numpy as np
from matplotlib import pyplot as plt
def plot_bar_and_text(x, y, width, label):
    plt.bar(x, y, width=width, label=label)
    for i in range(x.shape[0]):
        plt.text(x[i], y[i] + 0.2, str(round(y[i], 2)), ha='center')

plt.style.use(['science', 'muted'])
fig = plt.figure(figsize=(8, 4))
csv_array = np.loadtxt('./docs/source/_static/tutorials/clock_driven/10_forward_pattern/exe_time.csv', delimiter=',', skiprows=1)
T = csv_array[:, 0]
T_str = []
for i in range(T.shape[0]):
    T_str.append(str(int(T[i])))

t_cuda_ms_f = csv_array[:, 3]
t_cuda_f = csv_array[:, 2]
t_python_f = csv_array[:, 1]
t_cuda_ms_fb = csv_array[:, 6]
t_cuda_fb = csv_array[:, 5]
t_python_fb = csv_array[:, 4]
x = np.arange(0, T.shape[0]) * 2
width = 0.3
plot_bar_and_text(x - width * 1.5, t_python_f, width=width, label='neuron.LIFNode (Naive PyTorch)')
plot_bar_and_text(x, t_cuda_f, width=width, label='cext.neuron.LIFNode')
plot_bar_and_text(x + width * 1.5, t_cuda_ms_f, width=width, label='cext.neuron.MultiStepLIFNode')
plt.title('Execution time of Running Forward with $2^{20}$ Neurons')
plt.xlabel('simulation duration T (step)')
plt.ylabel('execution time (ms)')
plt.xticks(x, T_str)
plt.legend(frameon=True)
plt.savefig('./docs/source/_static/tutorials/clock_driven/10_forward_pattern/exe_time_f.svg')
plt.savefig('./docs/source/_static/tutorials/clock_driven/10_forward_pattern/exe_time_f.pdf')
plt.clf()
plot_bar_and_text(x - width * 1.5, t_python_fb, width=width, label='neuron.LIFNode (Naive PyTorch)')
plot_bar_and_text(x, t_cuda_fb, width=width, label='cext.neuron.LIFNode')
plot_bar_and_text(x + width * 1.5, t_cuda_ms_fb, width=width, label='cext.neuron.MultiStepLIFNode')
plt.title('Execution time of Running Forward and Backward with $2^{20}$ Neurons')
plt.xlabel('simulation duration T (step)')
plt.ylabel('execution time (ms)')
plt.xticks(x, T_str)
plt.legend(frameon=True)
plt.savefig('./docs/source/_static/tutorials/clock_driven/10_forward_pattern/exe_time_fb.svg')
plt.savefig('./docs/source/_static/tutorials/clock_driven/10_forward_pattern/exe_time_fb.pdf')