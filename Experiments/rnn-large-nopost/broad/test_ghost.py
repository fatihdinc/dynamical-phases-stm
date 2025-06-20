import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import src.data as data
import src.rnn as rnn
import src.data as data
import src.rnn as rnn
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

# Define the training function
def train_rnn_model(process_idx,delay_interval,learning_rate,sh=0,sv=1):
    import warnings
    warnings.filterwarnings('ignore')
    file_name =f'./results_nopost/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/class.txt'
    
        
      # Load the image
    f = np.load(f'./results_nopost/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/training_details.npz')
    acc = f['reaction_accuracy_list']
    rel = f['reaction_reliability_list']
    loss = f['loss_list']
    
    
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    in_dim, hid_dim, out_dim = 2, 100, 2
     # Example value
    noise_sigma = 0.001
    firing_rate_sigma = 0.1
    truncate_step = None
    input_interval = 0.030
    reaction_interval = 0.050
    
    delta_t = 5e-3
    time_decay = 10e-3
    
    alpha = delta_t / time_decay
    # Data Initialization
    dcdt = data.DCDT(
        input_interval=input_interval,
        delay_interval=delay_interval,
        reaction_interval=reaction_interval,
        post_reaction_interval=0,
        num_of_samples_per_each_class=64,
        delta_t=delta_t
    )

    dataloader = DataLoader(dcdt, batch_size=len(dcdt), shuffle=False)

    # Model Initialization
    model = rnn.RNN(
        in_dim=in_dim,
        hid_dim=hid_dim,
        out_dim=out_dim,
        alpha=alpha,
        noise_sigma=noise_sigma,
        firing_rate_sigma=firing_rate_sigma,
        truncate_step=truncate_step
    ).to(device)
    
    model.load_state_dict(torch.load(f'./results_nopost/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/model_final.pt', weights_only=True))
    

        
    extended_duration = 5 * dcdt.dataset.shape[2]
    extended_inputs = torch.zeros((dcdt.dataset.shape[0],2, extended_duration, dcdt.dataset.shape[-1])).to(device)
    # Correctly assign the original data to the extended input
    extended_inputs[:, :,:dcdt.dataset.shape[2], :] = dcdt.dataset[:, :,:, :].to(device)

    with torch.no_grad():
        out, fr = model(extended_inputs[:,0,:,:])
    
    oscillations = torch.sqrt(torch.mean((out[:,-dcdt.dataset.shape[2]:,:] - torch.mean(out[:,-dcdt.dataset.shape[2]:,:],1,keepdims = True))**2)) > 1e-1
    
    
    #plt.subplot(131)
    #plt.plot(acc)
    #plt.subplot(132)
    #plt.plot(rel)
    #plt.subplot(133)
    #plt.plot(out[0,:int(delay_interval/0.02)*5+100,0].detach().cpu().numpy())
    #plt.plot(out[0,:int(delay_interval/0.02)*5+100,1].detach().cpu().numpy())
    #plt.axvline(int(delay_interval/0.02)*5+16)
    
    
    ind = np.where( ((acc > 0.8) & (rel>0.8) ) )[0]
    if len(ind)==0:
        num = 1
    else:
        num = (np.sum( acc[ind[0]:]<0.6 ) + np.sum( rel[ind[0]:]<0.6 )) < 300
    
    
    
    
    acc = np.mean(acc[-100:])>0.7
    rel = np.mean(rel[-100:])>0.7
    
    learned = acc*rel*num
    
    if learned == 0:
        if num == 1:
            class_val = 0;
            #plt.title('no learn')
        else:
            class_val = 3;
            #plt.title('unstable learning')
        
    elif oscillations == 1:
        class_val = 1;
        #plt.title('LC')
    else:
        class_val = 2
        #plt.title('SP')
   
    
    
    
    #plt.close()
        
    
    # Write the input to the file
    with open(file_name, "w") as file:
        file.write(f"{class_val}")
        
    

lr_all = np.logspace(-1,-3,15);

lr_all = lr_all[:-3]

# Number of processes to run
n_processes = 4
n_exp = 100

classes = np.zeros([lr_all.shape[0],21,n_exp])

T_all = np.zeros(21)

for k in reversed(range(lr_all.shape[0])):
    learning_rate = lr_all[k]
    for l in tqdm((range(21))):
        delay_interval = l*0.020
        T_all[l] = delay_interval
        #for i in range(n_exp):
        #Parallel(n_jobs=n_processes)(delayed(train_rnn_model)(i,delay_interval,learning_rate) for i in range(n_exp))
        for i in range(n_exp):
            file_name =f'./results_nopost/lr{learning_rate}_delay_{delay_interval}/process_{i}/class.txt'
            f = open(file_name, "r")
            classes[k,l,i] = f.read()
            #train_rnn_model(i,delay_interval,learning_rate) 
            #continue
    plt.plot(np.mean(classes[k,:,:] == 0,1)+np.mean(classes[k,:,:] == 3,1))
    plt.plot(np.mean(classes[k,:,:] == 1,1))
    plt.plot(np.mean(classes[k,:,:] == 2,1))
    plt.title(learning_rate)
    plt.show()



#%%
import colorsys
classes[classes==3] = 0

met = np.zeros([lr_all.shape[0],21,3])
for i in range(lr_all.shape[0]):
    for j in range(21):
        temp = classes[i,j,:];
        
        met[i,j,1] = np.sum(temp == 1) / np.sum(temp > -1)
        met[i,j,2] = np.sum(temp == 2) / np.sum(temp > -1)
        met[i,j,0] = (np.sum(temp == 0) +np.sum(temp == 3)) / np.sum(temp > -1)
        
def rotate_hue(rgb, shift=0.05):  # Shift between 0 and 1 (0.2 is ~72 degrees)
    hsv = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in rgb.reshape(-1, 3)])
    hsv[:, 0] = (hsv[:, 0] + shift) % 1  # Rotate hue
    rgb_rotated = np.array([colorsys.hsv_to_rgb(*pixel) for pixel in hsv]).reshape(rgb.shape)
    return rgb_rotated


plt.imshow(rotate_hue(met),aspect = 'auto')
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11],np.round(np.log10(lr_all),2));
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20][::2],T_all[::2])
plt.show()

#%%
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for k in range(12):
    plt.plot(T_all,met[k,:,2],color = colors[0])
    plt.plot(T_all,met[k,:,0],color = colors[1])
    plt.plot(T_all,met[k,:,1],color = colors[2])
plt.show()

#%%


from scipy.optimize import curve_fit

# Define the function to fit
def func(x, a):
    return 1 / (1 + x / a)


def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))
# Define a piecewise linear function where x > 0 leads to a constant value
def relu(x, x0, y0, k1):
    return np.piecewise(x, [x >= x0, x < x0], [lambda x: k1 * (x - x0) + y0, lambda x: y0])

def f_scale(x,a,b):
    return a*x**b

a_all = np.zeros(12);

for k in range(12):
    temp = met[k,:,2]
    
    # Fit the function to the data
    popt, pcov = curve_fit(func, T_all, temp, p0=[0.1])  # Initial guess for a
    a_fit = popt[0]
    
    
    T_new = np.linspace(T_all.min(),T_all.max(),100)
    
    # Plot the data and the fitted function
    plt.scatter(T_all, temp, label='Noisy Data', color='red', alpha=0.6)
    plt.plot(T_new, func(T_new, *popt), label=f'Fitted Function (a={a_fit:.3f})', color='blue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
    a_all[k] = a_fit
    
    
    

#%%

import scipy.stats as stats


# Remove zeros to avoid log(0) issues
mask = (a_all[:6] > 0) & (lr_all[:6] > 0)
log_x = np.log(a_all[:6][mask])
log_y = np.log(lr_all[:6][mask])
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
plt.scatter(a_all, lr_all, label='Noisy Data', color='black')
plt.plot(x_range, y_pred, ls='-', color='blue')

# Plot confidence interval
plt.fill_between(x_range, y_lower, y_upper, color='blue', alpha=0.3, label='95% Confidence Interval')

# Log scale
plt.xscale('log')
plt.yscale('log')

plt.title(f" Critical exponent: {b_fit:.2f} ± {b_err:.2f}, R2: {r2:.2f}, p {p_value:.4f}")


#plt.savefig('scaling.pdf')
plt.show()


