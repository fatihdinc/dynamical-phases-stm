import numpy as np
import matplotlib.pyplot as plt



T_resp = 4
T_delays = np.linspace(8,400,100)

scaling_ghost = np.pi**4/4 * T_resp * T_delays**-5
scaling_lc = T_resp * (T_delays**-3) / 4

scaling_ghost_eq = 3*np.pi**4/32 * T_delays**-4
scaling_lc_eq =  (T_delays**-2)/24 



plt.subplot(121)
plt.semilogy(T_delays,scaling_ghost,color = '#1f77b4')
plt.semilogy(T_delays,scaling_lc,color = '#ff7f0e')
plt.ylim([1e-9,1e-2])
plt.subplot(122)
plt.semilogy(T_delays,scaling_ghost_eq,color = '#1f77b4',ls = '--')
plt.semilogy(T_delays,scaling_lc_eq,color = '#ff7f0e',ls = '--')
plt.ylim([1e-9,1e-2])