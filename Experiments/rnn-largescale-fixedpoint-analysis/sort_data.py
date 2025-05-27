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
    file_name =f'./results_fixed/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/class.txt'
    
        
      # Load the image
    f = np.load(f'./results_fixed/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/training_details.npz')
    acc = f['reaction_accuracy_list']
    rel = f['reaction_reliability_list']

    
    
    start = int(acc.shape[0]*0.9)
    acc = np.mean(acc[start:])>0.8
    rel = np.mean(rel[start:])>0.8
    learned = acc*rel
    
    
    if learned == 0:
        
        class_val = 0;
      
    else:

   
        class_val = 1
   
    
    
    #plt.close()
        
    
    # Write the input to the file
    with open(file_name, "w") as file:
        file.write(f"{class_val}")
        
    

lr_all = np.logspace(0,-3,20);
# Number of processes to run
n_processes = 4
n_exp = 20

classes = np.zeros([lr_all.shape[0],20,n_exp])

T_all = np.zeros(20)

for k in reversed(range(lr_all.shape[0])):
    learning_rate = lr_all[k]
    for l in tqdm((range(20))):
        delay_interval = (l+1)*0.020
        T_all[l] = delay_interval
        #for i in range(n_exp):
        #Parallel(n_jobs=n_processes)(delayed(train_rnn_model)(i,delay_interval,learning_rate) for i in range(n_exp))
        for i in range(n_exp):
            file_name =f'./results_fixed/lr{learning_rate}_delay_{delay_interval}/process_{i}/class.txt'
            f = open(file_name, "r")
            classes[k,l,i] = f.read()
            #train_rnn_model(i,delay_interval,learning_rate) 
            #continue
    plt.plot(np.mean(classes[k,:,:] == 0,1))
    plt.plot(np.mean(classes[k,:,:] == 1,1))
    plt.title(learning_rate)
    plt.show()


#%%
import colorsys

met = np.zeros([lr_all.shape[0],20,3])
for i in range(lr_all.shape[0]):
    for j in range(20):
        temp = classes[i,j,:];
        
        met[i,j,2] = 0
        met[i,j,0] = np.sum(temp == 0) / np.sum(temp >-1)
        met[i,j,1] = np.sum(temp == 1) / np.sum(temp >-1)
        
        
def rotate_hue(rgb, shift=0):  # Shift between 0 and 1 (0.2 is ~72 degrees)
    hsv = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in rgb.reshape(-1, 3)])
    hsv[:, 0] = (hsv[:, 0] + shift) % 1  # Rotate hue
    rgb_rotated = np.array([colorsys.hsv_to_rgb(*pixel) for pixel in hsv]).reshape(rgb.shape)
    return rgb_rotated


plt.imshow(rotate_hue(met),aspect = 'auto')
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],np.round(np.log10(lr_all),2));
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20][::2],T_all[1::2])
plt.savefig('phase_diagram.pdf')
plt.show()


#%%








