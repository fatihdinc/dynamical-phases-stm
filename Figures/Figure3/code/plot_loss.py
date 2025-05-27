import numpy as np
import matplotlib.pyplot as plt

f = np.load('post.npz')
post = f['loss_history']
f.close()

f = np.load('low_T.npz')
low = f['loss_history']
f.close()


post = post/ post[0]

low = low/low[0]

x = np.linspace(1,low.shape[0],low.shape[0])
plt.loglog(x,low/low[0])

x = np.linspace(1,post.shape[0],post.shape[0])
plt.loglog(x,post/post[0])

plt.title('2.71 * 1e-5')
plt.savefig('loss.pdf')
