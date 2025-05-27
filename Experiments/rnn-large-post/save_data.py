import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

lr_all = np.logspace(-1,-3,15);
# Number of processes to run
n_processes = 4
n_exp = 20

classes = np.zeros([lr_all.shape[0],21,n_exp]) + np.nan
losses = np.zeros([lr_all.shape[0],21,n_exp,1000]) + np.nan
rels = np.zeros([lr_all.shape[0],21,n_exp,1000]) + np.nan
accs = np.zeros([lr_all.shape[0],21,n_exp,1000]) + np.nan
psd_ratios = np.zeros([lr_all.shape[0],21,n_exp]) + np.nan
T_all = np.zeros(21)

for k in range(lr_all.shape[0]):
    learning_rate = lr_all[k]
    for l in tqdm((range(21))):
        delay_interval = l*0.020
        T_all[l] = delay_interval
        for i in range(n_exp):
            file_name =f'./results_post/lr{learning_rate}_delay_{delay_interval}/process_{i}/class.txt'
            f = open(file_name, "r")
            classes[k,l,i] = f.read()
            f.close()
            
            
            if ( (classes[k,l,i] == 1) or  (classes[k,l,i] == 2) ):
            
                file_name =f'./results_post/lr{learning_rate}_delay_{delay_interval}/process_{i}/psd_ratio.txt'
                f = open(file_name, "r")
                psd_ratios[k,l,i] = f.read()
                f.close()
            
            
            
            f = np.load(f'./results_post/lr{learning_rate}_delay_{delay_interval}/process_{i}/training_details.npz')
            acc = f['reaction_accuracy_list']
            rel = f['reaction_reliability_list']
            loss = f['loss_list']

            sz = acc.shape[0]
            if sz<1000:
                rels[k,l,i,-sz:]=rel[-sz:]
                accs[k,l,i,-sz:]=acc[-sz:]
                losses[k,l,i,-sz:]=loss[-sz:]
            else:
                sz = 1000
                rels[k,l,i,-sz:]=rel[-sz:]
                accs[k,l,i,-sz:]=acc[-sz:]
                losses[k,l,i,-sz:]=loss[-sz:]
            
    plt.plot(np.mean(classes[k,:,:] == 0,1))
    plt.plot(np.mean(classes[k,:,:] == 1,1))
    plt.plot(np.mean(classes[k,:,:] == 2,1))
    plt.plot(np.mean(classes[k,:,:] == 3,1))
    plt.title(learning_rate)
    plt.show()
    
np.savez('post.npz',classes=classes,losses=losses,rels=rels,accs=accs,psd_ratios=psd_ratios)

#%%


for lr_pick in range(30):
    print(lr_all[lr_pick])
    
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for k in [0,1,2,3]:
            
        #plt.subplot(4,1,k+1)
        ind = np.where( (classes[lr_pick,:,:] == k))
        temp = losses[lr_pick,:,:,:];
        temp = temp[ind]
        
        plt.errorbar(np.linspace(1,10000,10000),np.nanmean(temp,0),np.nanstd(temp,0)/np.sqrt(temp.shape[0]),color = colors[k])
        
        ## Select 20 random indices instead of the first 20
        #num_samples = min(100, temp.shape[0])  # Ensure we don't exceed available samples
        #random_indices = np.random.choice(temp.shape[0], num_samples, replace=False)
        #temp = temp[random_indices, :]
        
        
        #plt.plot(temp.T,color = colors[k],linewidth = .1)
        
    plt.yscale('log')
    plt.show()



