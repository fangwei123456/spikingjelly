import torch
from matplotlib import pyplot as plt
def reset_v(h, s):
    return h * (1 - s)
x = torch.arange(-1, 1.01, 0.01)
figure = plt.figure(dpi=200)
fig0 = plt.subplot(1, 2, 1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$\\Theta(x)$ and $\\sigma(\\alpha x)$')
plt.plot(x, (x >= 0).float(), label='$\\Theta(x)$')
plt.plot(x, torch.sigmoid(5 * x), linestyle=':', label='$\\sigma(\\alpha x), \\alpha=5.0$')
plt.plot(x, torch.sigmoid(10 * x), linestyle=':', label='$\\sigma(\\alpha x), \\alpha=10.0$')
plt.plot(x, torch.sigmoid(50 * x), linestyle=':', label='$\\sigma(\\alpha x), \\alpha=50.0$')
plt.legend()

fig1 = plt.subplot(1, 2, 2)
h = torch.arange(0, 2.5, 0.01)

plt.xlabel('$H(t)$')
plt.ylabel('$V(t)$')
plt.title('Voltage Reset')
plt.plot(h, reset_v(h, (h >= 1).float()), label='$\\Theta(x)$')
plt.plot(h, reset_v(h, torch.sigmoid(5 * (h - 1))), linestyle=':', label='$\\sigma(\\alpha x), \\alpha=5.0$')
plt.plot(h, reset_v(h, torch.sigmoid(10 * (h - 1))), linestyle=':', label='$\\sigma(\\alpha x), \\alpha=10.0$')
plt.plot(h, reset_v(h, torch.sigmoid(50 * (h - 1))), linestyle=':', label='$\\sigma(\\alpha x), \\alpha=50.0$')
plt.axhline(0, linestyle='--', label='$V_{reset}$', c='g')
plt.axhline(1, linestyle='--', label='$V_{threshold}$', c='r')

plt.legend()
plt.show()