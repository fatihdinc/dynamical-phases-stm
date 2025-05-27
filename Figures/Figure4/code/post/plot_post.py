#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:49:04 2025

@author: dinc
"""
import numpy as np
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
lr_all = np.logspace(-1,-3,15)

T_all = np.linspace(0,400,21)

f = np.load('post.npz')

losses = np.array(f['losses'])
classes = np.array(f['classes'])
rels=np.array(f['rels'])
accs=np.array(f['accs'])
psd_ratio=np.array(f['psd_ratios'])
lr_pick = 0
print(lr_all[lr_pick])

classes[classes == 3] = 0

metric = losses.copy()
for k in [0,1,2]:
    ind = np.where( (classes[lr_pick,:,:] == k))
    temp = metric[lr_pick,:,:,:];
    temp = temp[ind]
    
    plt.errorbar(np.linspace(1,1000,1000),np.nanmean(temp,0),np.nanstd(temp,0)/np.sqrt(temp.shape[0]),color = colors[k])

plt.savefig('losses_post.pdf')
plt.show()



#%%
import colorsys
#classes[classes==3] = 0

met = np.zeros([lr_all.shape[0],21,3])
for i in range(lr_all.shape[0]):
    for j in range(21):
        temp = classes[i,j,:];
        
        met[i,j,0] = (np.sum(temp == 0)+np.sum(temp == 3) ) / np.sum(temp >-1)
        met[i,j,1] = np.sum(temp == 1) / np.sum(temp >-1)
        met[i,j,2] = np.sum(temp == 2) / np.sum(temp >-1)
        

for k in range(lr_all.shape[0]):
    plt.plot(T_all,met[k,:,0],color = colors[0])
    plt.plot(T_all,met[k,:,1],color = colors[1])
    plt.plot(T_all,met[k,:,2],color = colors[2])
plt.savefig('different_strategies.pdf')
plt.show()

#%%
st = 0
from scipy.optimize import curve_fit
# Define the function to fit
def func(x, a):
    return 1 / (1 + x / a)

def exp_saturation(x, x0, a):
    """
    Computes a piecewise function that is 0 before x0 and follows 
    an exponential saturation to 1 for x >= x0.
    
    Parameters:
    x  : float or numpy array - input value(s)
    x0 : float - threshold where function starts increasing
    a  : float - growth rate of the exponential function
    
    Returns:
    f(x) : float or numpy array - evaluated function values
    """
    return np.where(x < x0, 0, 1 - np.exp(-a * (x - x0)))


def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))
# Define a piecewise linear function where x > 0 leads to a constant value
def relu(x, x0, y0, k1):
    return np.piecewise(x, [x >= x0, x < x0], [lambda x: k1 * (x - x0) + y0, lambda x: y0])

def f_scale(x,a,b):
    return a*x**b

a_all = np.zeros(lr_all.shape[0]);
for k in range(lr_all.shape[0]):
    
    temp = met[k,:,st]
    

    if np.max(temp)<0.2:
        a_all[k] = 0
        continue
    
    
    
    # Fit the function to the data
    if st == 0:
        popt, pcov = curve_fit(exp_saturation, T_all, temp, p0=[0.5,0.1])  # Initial guess for a
    else:
        popt, pcov = curve_fit(sigmoid, T_all, temp, p0=[0.1,0.1])  # Initial guess for a
    a_fit = popt[0]#+1/popt[1]
    #try:
    #    a_fit = T_all[np.where(temp>0.1)[0][0]] #popt[0]
    #except:
    #    a_fit = 0
    T_new = np.linspace(T_all.min(),T_all.max(),100)
    
    # Plot the data and the fitted function
    plt.scatter(T_all, temp, label='Noisy Data', color=colors[st], alpha=0.6)
    if st ==0:
        plt.plot(T_new, exp_saturation(T_new, *popt), label=f'Fitted Function (a={1000*a_fit:.3f})', color=colors[st])
    else:
        plt.plot(T_new, sigmoid(T_new, *popt), label=f'Fitted Function (a={1000*a_fit:.3f})', color=colors[st])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim([0,1])
    plt.show()
    a_all[k] = a_fit
    
    
#%%

import scipy.stats as stats

ns = 100
# Remove zeros to avoid log(0) issues
mask = (a_all[:ns] > 0) & (lr_all[:ns] > 0)
mask[1] = 0
log_x = np.log(a_all[:ns][mask])
log_y = np.log(lr_all[:ns][mask])
# Define linear function for log-log fit
def linear_model(log_a, B, log_A):
    return B * log_a + log_A  # Equivalent to log(y) = B log(x) + log(A)

# Fit log-log data with a linear model
popt, pcov = curve_fit(linear_model, log_x, log_y)


# Perform linear regression in log-log space
slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
r2 = r_value ** 2

# Extract fit parameters
b_fit, log_a_fit = popt  # log(A) is the intercept
a_fit = np.exp(log_a_fit)  # Convert back to original scale

# Compute standard errors correctly
perr = np.sqrt(np.diag(pcov))  # Errors in log parameters
b_err = perr[0]  # Error in slope B


# Compute fitted values in log space
log_x_range = np.linspace(min(log_x), max(log_x), 100)  # Smooth log-space
log_y_pred = linear_model(log_x_range, b_fit, log_a_fit)

# Convert back to original scale
x_range = np.exp(log_x_range)
y_pred = np.exp(log_y_pred)

# Generate Monte Carlo samples for confidence intervals
num_samples = 1000
param_samples = np.random.multivariate_normal(popt, pcov, size=num_samples)

# Compute confidence intervals
y_samples = np.array([linear_model(log_x_range, B, log_A) for B, log_A in param_samples])
y_lower = np.exp(np.percentile(y_samples, 2.5, axis=0))
y_upper = np.exp(np.percentile(y_samples, 97.5, axis=0))

# Plot data

plt.plot(x_range, y_pred, ls='-', color=colors[st])

# Plot confidence interval
plt.fill_between(x_range, y_lower, y_upper, color=colors[st], alpha=0.5)
plt.scatter(a_all, lr_all,20, color='black')

# Log scale
plt.xscale('log')
plt.yscale('log')
plt.ylim([8e-4,5e-1])
plt.xlim([30,300])

plt.title(f" Critical exponent: {b_fit:.2f} ± {b_err:.2f}, R2: {r2:.2f}, p {p_value:.4f}")
plt.savefig('scaling.pdf')
plt.show()


#%%
st = 2
from scipy.optimize import curve_fit
# Define the function to fit
def func(x, a):
    return 1 / (1 + x / a)

def exp_saturation(x, x0, a):
    """
    Computes a piecewise function that is 0 before x0 and follows 
    an exponential saturation to 1 for x >= x0.
    
    Parameters:
    x  : float or numpy array - input value(s)
    x0 : float - threshold where function starts increasing
    a  : float - growth rate of the exponential function
    
    Returns:
    f(x) : float or numpy array - evaluated function values
    """
    return np.where(x < x0, 0, 1 - np.exp(-a * (x - x0)))


def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))
# Define a piecewise linear function where x > 0 leads to a constant value
def relu(x, x0, y0, k1):
    return np.piecewise(x, [x >= x0, x < x0], [lambda x: k1 * (x - x0) + y0, lambda x: y0])

def f_scale(x,a,b):
    return a*x**b

a_all = np.zeros(lr_all.shape[0]);
for k in range(lr_all.shape[0]):
    
    temp = met[k,:,st]
    

    if np.max(temp)<0.2:
        a_all[k] = 0
        continue
    
    
    # Fit the function to the data
    if st == 0:
        popt, pcov = curve_fit(exp_saturation, T_all, temp, p0=[0.5,0.1])  # Initial guess for a
    else:
        popt, pcov = curve_fit(sigmoid, T_all, temp, p0=[0.1,0.1])  # Initial guess for a
    a_fit = popt[0]#+1/popt[1]
    #try:
    #    a_fit = T_all[np.where(temp>0.1)[0][0]] #popt[0]
    #except:
    #    a_fit = 0
    T_new = np.linspace(T_all.min(),T_all.max(),100)
    
    # Plot the data and the fitted function
    plt.scatter(T_all, temp, label='Noisy Data', color=colors[st], alpha=0.6)
    if st ==0:
        plt.plot(T_new, exp_saturation(T_new, *popt), label=f'Fitted Function (a={1000*a_fit:.3f})', color=colors[st])
    else:
        plt.plot(T_new, sigmoid(T_new, *popt), label=f'Fitted Function (a={1000*a_fit:.3f})', color=colors[st])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim([0,1])
    plt.show()
    a_all[k] = a_fit
    
    
#%%

import scipy.stats as stats

ns = 100
# Remove zeros to avoid log(0) issues
mask = (a_all[:ns] > 0) & (lr_all[:ns] > 0)
mask[1] = 0
log_x = np.log(a_all[:ns][mask])
log_y = np.log(lr_all[:ns][mask])
# Define linear function for log-log fit
def linear_model(log_a, B, log_A):
    return B * log_a + log_A  # Equivalent to log(y) = B log(x) + log(A)

# Fit log-log data with a linear model
popt, pcov = curve_fit(linear_model, log_x, log_y)


# Perform linear regression in log-log space
slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
r2 = r_value ** 2

# Extract fit parameters
b_fit, log_a_fit = popt  # log(A) is the intercept
a_fit = np.exp(log_a_fit)  # Convert back to original scale

# Compute standard errors correctly
perr = np.sqrt(np.diag(pcov))  # Errors in log parameters
b_err = perr[0]  # Error in slope B


# Compute fitted values in log space
log_x_range = np.linspace(min(log_x), max(log_x), 100)  # Smooth log-space
log_y_pred = linear_model(log_x_range, b_fit, log_a_fit)

# Convert back to original scale
x_range = np.exp(log_x_range)
y_pred = np.exp(log_y_pred)

# Generate Monte Carlo samples for confidence intervals
num_samples = 1000
param_samples = np.random.multivariate_normal(popt, pcov, size=num_samples)

# Compute confidence intervals
y_samples = np.array([linear_model(log_x_range, B, log_A) for B, log_A in param_samples])
y_lower = np.exp(np.percentile(y_samples, 2.5, axis=0))
y_upper = np.exp(np.percentile(y_samples, 97.5, axis=0))

# Plot data

plt.plot(x_range, y_pred, ls='-', color=colors[st])

# Plot confidence interval
plt.fill_between(x_range, y_lower, y_upper, color=colors[st], alpha=0.5)
plt.scatter(a_all, lr_all,20, color='black')

# Log scale
plt.xscale('log')
plt.yscale('log')
plt.ylim([8e-4,5e-1])
plt.xlim([30,300])

plt.title(f" Critical exponent: {b_fit:.2f} ± {b_err:.2f}, R2: {r2:.2f}, p {p_value:.4f}")
plt.savefig('scaling_sp.pdf')
plt.show()
