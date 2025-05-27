import numpy as np
import matplotlib.pyplot as plt

# Define the function
def L(r):
    if -1 / (2 * (T_delay + T_reac)) <= r <= 1 / (2 * (T_delay + T_reac)):
        return 1
    elif 1 / (2 * (T_delay + T_reac)) < r <= 1 / (2 * T_delay):
        return 1 / (2 * T_reac*r)  - T_delay / T_reac
    elif 1 / (2 * T_delay) < r <= 1 / (T_delay + T_reac):
        return 1 - 1 / (2 * T_delay * r)
    elif 1 / (T_reac+ T_delay) < r <= 1 / (T_delay):
        return 2+T_delay/T_reac -1/2/r/T_delay - 1/r/T_reac
    elif (T_delay == T_reac) and (1/2/T_delay <= r <= 3/4/T_delay):
        return 3- 3/T_delay/r/2
    else:
        return np.nan  # Return NaN for out-of-bound values

def loss_function(r, T_delay, T_reac,dt = 0.01):
    
    x = np.linspace(0,T_delay,int(T_delay/dt))
    
    # First integral term
    integral_1 = dt / T_delay * np.sum(np.heaviside(-np.sin(2 * np.pi * r *x), 0))
    
    x = np.linspace(T_delay,T_delay+T_reac,int(T_reac/dt))
    # Second integral term
    integral_2 = dt / T_reac * np.sum(1 - np.heaviside(-np.sin(2 * np.pi * r * x), 0))
    
    # Loss
    return integral_1 + integral_2

# Parameter values
T_delay = 60  # Example value, adjust as needed
T_reac = 10  # Example value, adjust as needed


# Create an array of r values
r_values = np.linspace(0 / (2 * (T_delay + T_reac)), 1 / (T_delay), 100)



L_values = [L(r) for r in r_values]



r_values_emp = np.linspace(0 / (2* (T_delay + T_reac)), 1 / (T_delay), 300)

#r_values_emp = np.linspace(0,0.002,1000)

loss_values = [loss_function(r, T_delay, T_reac) for r in r_values_emp]

plt.plot(r_values,L_values,color = 'black',linewidth =2)

plt.scatter(r_values_emp, loss_values)
plt.xlabel("r")
plt.ylabel(r"$\mathcal{L}(r)$")
plt.title(f"T_delay={T_delay}, T_reac={T_reac}")

plt.savefig('loss_function.pdf')


