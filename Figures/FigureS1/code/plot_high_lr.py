import numpy as np
import matplotlib.pyplot as plt

f = np.load('high_lr.npz')



grad_history = f['grad_history']
loss_history = f['loss_history']

x = np.linspace(1,150000,150000)

plt.semilogy(x[::30],loss_history[::30])
plt.axvline(40000)
plt.axvline(67000)
plt.axvline(130000)
plt.savefig('loss_high_lr.pdf')
