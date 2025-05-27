import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


lr_all = np.logspace(0,-3,20);
# Number of processes to run
n_processes = 4
n_exp = 20

classes = np.zeros([lr_all.shape[0],20,n_exp]) + np.nan
losses = np.zeros([lr_all.shape[0],20,n_exp,3000]) + np.nan
rels = np.zeros([lr_all.shape[0],20,n_exp,3000]) + np.nan
accs = np.zeros([lr_all.shape[0],20,n_exp,3000]) + np.nan



T_all = np.zeros(20)

for k in range(lr_all.shape[0]):
    learning_rate = lr_all[k]
    for l in tqdm((range(20))):
        delay_interval = (l+1)*0.020
        T_all[l] = delay_interval
        for i in range(n_exp):
            file_name =f'./results_fixed/lr{learning_rate}_delay_{delay_interval}/process_{i}/class.txt'
            f = open(file_name, "r")
            classes[k,l,i] = f.read()
            f.close()

            
            
            f = np.load(f'./results_fixed/lr{learning_rate}_delay_{delay_interval}/process_{i}/training_details.npz')
            acc = f['reaction_accuracy_list']
            rel = f['reaction_reliability_list']
            loss = f['loss_list']

            sz = acc.shape[0]
            if sz<3000:
                rels[k,l,i,:sz]=rel[:sz]
                accs[k,l,i,:sz]=acc[:sz]
                losses[k,l,i,:sz]=loss[:sz]
            else:
                sz = 3000
                rels[k,l,i,-sz:]=rel[-sz:]
                accs[k,l,i,-sz:]=acc[-sz:]
                losses[k,l,i,-sz:]=loss[-sz:]
            
    plt.plot(np.mean(classes[k,:,:] == 0,1))
    plt.plot(np.mean(classes[k,:,:] == 1,1))
    plt.title(learning_rate)
    plt.show()
    
np.savez('fixed.npz',classes=classes,losses=losses,rels=rels,accs=accs)



