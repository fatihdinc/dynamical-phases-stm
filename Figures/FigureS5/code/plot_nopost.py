#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:49:04 2025

@author: dinc
"""
import numpy as np
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
lr_all =  np.logspace(-1,-3,10);

T_all = np.linspace(0,800,21)

f = np.load('longtau.npz')

losses = np.array(f['losses'])
classes = np.array(f['classes'])
rels=np.array(f['rels'])
accs=np.array(f['accs'])
psd_ratio=np.array(f['psd_ratios'])
lr_pick = 0
print(lr_all[lr_pick])

metric = losses.copy()
for k in [0,1,2,3]:
    ind = np.where( (classes[lr_pick,:,:] == k))
    temp = metric[lr_pick,:,:,:];
    temp = temp[ind]
    
    plt.errorbar(np.linspace(1,1000,1000),np.nanmean(temp,0),np.nanstd(temp,0)/np.sqrt(temp.shape[0]),color = colors[k])

plt.savefig('losses.pdf')
plt.show()


metric = accs.copy()
for k in [0,1,2,3]:
    ind = np.where( (classes[lr_pick,:,:] == k))
    temp = metric[lr_pick,:,:,:];
    temp = temp[ind]
    
    plt.errorbar(np.linspace(1,1000,1000),np.nanmean(temp,0),np.nanstd(temp,0)/np.sqrt(temp.shape[0]),color = colors[k])

plt.ylim([0.49,1])
plt.savefig('accs.pdf')
plt.show()


metric = rels.copy()
for k in [0,1,2,3]:
    ind = np.where( (classes[lr_pick,:,:] == k))
    temp = metric[lr_pick,:,:,:];
    temp = temp[ind]
    
    plt.errorbar(np.linspace(1,1000,1000),np.nanmean(temp,0),np.nanstd(temp,0)/np.sqrt(temp.shape[0]),color = colors[k])

plt.ylim([0.49,1])
plt.savefig('rels.pdf')
plt.show()



#%%
from scipy.stats import norm, expon

ind = np.where(classes == 1)
a = psd_ratio[ind].flatten()
ind = np.where(classes == 2)
b = psd_ratio[ind].flatten()

# Fit Gaussians
lambda_a = 1 / np.mean(a)
mu_b, std_b = norm.fit(b)


plt.hist(a,bins = np.linspace(0,1,50),color = colors[1],density = True)
plt.hist(b,bins = np.linspace(0,1,50),density = True,color = colors[2])

# Define range for plotting Gaussians
x = np.linspace(-0, 1, 100)

# Gaussian PDFs
pdf_b = norm.pdf(x, mu_b, std_b)
pdf_a = expon.pdf(x, scale=1/lambda_a)

plt.plot(x, pdf_a, color=colors[1], linestyle='--', label='LC')
plt.plot(x, pdf_b, color=colors[2], linestyle='--', label='SP')
plt.legend()
plt.savefig('PSD_ratios.pdf')
plt.show()


#%%
import colorsys
#classes[classes==3] = 0

st = 0

met = np.zeros([lr_all.shape[0],21,3])
for i in range(lr_all.shape[0]):
    for j in range(21):
        temp = classes[i,j,:];
        
        met[i,j,0] = (np.sum(temp == 0)+np.sum(temp == 3) ) / np.sum(temp >-1)
        met[i,j,1] = np.sum(temp == 1) / np.sum(temp >-1)
        met[i,j,2] = np.sum(temp == 2) / np.sum(temp >-1)

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

count = 0
for k in [0,4,9]:
    
    temp = met[k,:,st]
    
    
    # Fit the function to the data
    if st == 0:
        popt, pcov = curve_fit(exp_saturation, T_all, temp, p0=[0.1,0.1])  # Initial guess for a
    else:
        popt, pcov = curve_fit(sigmoid, T_all, temp, p0=[0.1,0.1])  # Initial guess for a
    a_fit = popt[0]#+1/popt[1]
    #try:
    #    a_fit = T_all[np.where(temp>0.1)[0][0]] #popt[0]
    #except:
    #    a_fit = 0
    T_new = np.linspace(T_all.min(),T_all.max(),100)
    
    # Plot the data and the fitted function
    plt.scatter(T_all, temp, label=np.round(lr_all[k],3), color=colors[count])
    if st ==0:
        plt.plot(T_new, exp_saturation(T_new, *popt), color=colors[count])
    else:
        plt.plot(T_new, sigmoid(T_new, *popt), color=colors[count])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim([0,1])
    count = count+1
plt.legend()
plt.savefig('example_fit_class0.pdf')
plt.show()
    
