import numpy as np
import matplotlib.pyplot as plt

f = np.load('results_lr_0.0001.npz')
loss = np.array(f['loss_history'])
grad = np.array(f['grad_history'])
f.close()


plot_every = 10

loss = loss/ loss[0]
grad = grad/grad[0] 

x = np.linspace(1,loss.shape[0],loss.shape[0])
plt.loglog(x[::plot_every],loss[::plot_every]/loss[0])

x = np.linspace(1,grad.shape[0],grad.shape[0])
plt.loglog(x[::plot_every],grad[::plot_every]/grad.max(),alpha = 0.5)

plt.savefig('loss.pdf')
