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
    file_name2 =f'./results_nopost/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/psd_ratio.txt'
    
    def createCircularMask(h, w, center, radius):

        X, Y = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
        mask = dist_from_center <= radius
        return mask
        
      # Load the image
    f = np.load(f'./results_nopost/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/training_details.npz')
    acc = f['reaction_accuracy_list']
    rel = f['reaction_reliability_list']
    loss = f['loss_list']
    
    ind = np.where( ((acc > 0.8) & (rel>0.8) ) )[0]
    if len(ind)==0:
        num = 1
    else:
        start = ind[0]
        num = ( (np.sum( acc[start:]<0.6 ) + np.sum( rel[start:]<0.6 )) / \
               (np.sum( acc[start:]>0.6 ) + np.sum( rel[start:]>0.6 )) ) < 0.1
    
    
    
    
    start = int(acc.shape[0]*0.9)
    acc = np.mean(acc[start:])>0.8
    rel = np.mean(rel[start:])>0.8
    
    learned = acc*rel*num
    
    if learned == 0:
        if num == 1:
            class_val = 0;
            #plt.title('no learn')
        else:
            class_val = 3;
            #plt.title('unstable learning')
        with open(file_name, "w") as file:
            file.write(f"{class_val}")
        return np.nan
    
    
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
    

        
    # extended_duration = 5 * dcdt.dataset.shape[2]
    extended_duration = 9 * dcdt.dataset.shape[2]
    extended_inputs = torch.zeros((dcdt.dataset.shape[0],2, extended_duration, dcdt.dataset.shape[-1])).to(device)
    # Correctly assign the original data to the extended input
    extended_inputs[:, :,:dcdt.dataset.shape[2], :] = dcdt.dataset[:, :,:, :].to(device)

    with torch.no_grad():
        out, fr = model(extended_inputs[:,0,:,:])
    
    #oscillations = torch.sqrt(torch.mean((out[:,-dcdt.dataset.shape[2]:,:] - torch.mean(out[:,-dcdt.dataset.shape[2]:,:],1,keepdims = True))**2)) > 1e-1
    
    U, S, V = torch.pca_lowrank(fr[0])
    low_rank = torch.matmul(fr[0], V[:, :2])

    # total = torch.sum(torch.corrcoef(low_rank))

    corr_coef = torch.corrcoef(low_rank)
    fft_signal = torch.fft.fft2(corr_coef)
    fft_signal_shifted = torch.fft.fftshift(fft_signal)
    magnitude = torch.abs(fft_signal_shifted)

    # total = torch.sum(magnitude)

    # magnitude[magnitude.shape[0]//2, magnitude.shape[1]//2] = 0

    total_psd = torch.sum(magnitude**2)
    outside_psd = torch.sum(magnitude[createCircularMask(magnitude.shape[0], magnitude.shape[1], [magnitude.shape[0]//2, magnitude.shape[1]//2], radius=3)]**2)


    ###
    #outside_psd/total_psd
    if (outside_psd/total_psd).item() < 0.5:
        class_val = 1
        #plt.title('LC')
    else:
        class_val = 2
        #plt.title('SP')
        
    
    # Write the input to the file
    with open(file_name, "w") as file:
        file.write(f"{class_val}")
        
    with open(file_name2, "w") as file:
        file.write(f"{(outside_psd / total_psd).item():.4f}")
        
    return class_val

lr_all = np.logspace(-1,-3,15);

# Number of processes to run
n_processes = 4
n_exp = 100

classes = np.zeros([lr_all.shape[0],21,n_exp])
psd_ratio = np.zeros([lr_all.shape[0],21,n_exp])

T_all = np.zeros(21)

for k in reversed(range(lr_all.shape[0])):
    learning_rate = lr_all[k]
    for l in tqdm((range(21))):
        delay_interval = l*0.020
        T_all[l] = delay_interval
        #Parallel(n_jobs=n_processes)(delayed(train_rnn_model)(i,delay_interval,learning_rate) for i in range(n_exp))
        for i in range(n_exp):
            file_name =f'./results_nopost/lr{learning_rate}_delay_{delay_interval}/process_{i}/class.txt'
            f = open(file_name, "r")
            classes[k,l,i] = f.read()
            f.close()
            
            if ( (classes[k,l,i] == 1) or  (classes[k,l,i] == 2) ):
            
                file_name =f'./results_nopost/lr{learning_rate}_delay_{delay_interval}/process_{i}/psd_ratio.txt'
                f = open(file_name, "r")
                psd_ratio[k,l,i] = f.read()
                f.close()
            
            
    plt.plot(np.mean(classes[k,:,:] == 0,1))
    plt.plot(np.mean(classes[k,:,:] == 1,1))
    plt.plot(np.mean(classes[k,:,:] == 2,1))
    plt.plot(np.mean(classes[k,:,:] == 3,1))
    plt.title(learning_rate)
    plt.show()

#%%
import colorsys
#classes[classes==3] = 0

met = np.zeros([lr_all.shape[0],21,3])
for i in range(lr_all.shape[0]):
    for j in range(21):
        temp = classes[i,j,:];
        
        met[i,j,2] = (np.sum(temp == 0)+np.sum(temp == 3) ) / np.sum(temp >-1)
        met[i,j,0] = np.sum(temp == 1) / np.sum(temp >-1)
        met[i,j,1] = np.sum(temp == 2) / np.sum(temp >-1)
        
        
def rotate_hue(rgb, shift=0):  # Shift between 0 and 1 (0.2 is ~72 degrees)
    hsv = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in rgb.reshape(-1, 3)])
    hsv[:, 0] = (hsv[:, 0] + shift) % 1  # Rotate hue
    rgb_rotated = np.array([colorsys.hsv_to_rgb(*pixel) for pixel in hsv]).reshape(rgb.shape)
    return rgb_rotated


plt.imshow(rotate_hue(met),aspect = 'auto')
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],np.round(np.log10(lr_all),2));
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20][::2],T_all[::2])
plt.savefig('phase_diagram.pdf')
plt.show()




