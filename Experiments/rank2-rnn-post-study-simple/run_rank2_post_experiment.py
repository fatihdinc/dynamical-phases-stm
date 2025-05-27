import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import os
from joblib import Parallel, delayed


class WeightedMSELoss(nn.Module):
    def __init__(self, class_weights):
        """
        Initializes the Weighted MSE Loss.
        
        Args:
            class_weights (torch.Tensor): A tensor of weights for each class.
        """
        super(WeightedMSELoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, predictions, targets):
        """
        Computes the weighted MSE loss.
        
        Args:
            predictions (torch.Tensor): The predicted outputs of the model. Shape: (batch_size, num_classes)
            targets (torch.Tensor): The ground truth outputs. Shape: (batch_size, num_classes)
        
        Returns:
            torch.Tensor: The weighted MSE loss.
        """
        # Compute the element-wise squared differences
        mse_loss = (predictions - targets) ** 2

        # Apply class weights
        weighted_loss = mse_loss * self.class_weights

        # Reduce the weighted loss by averaging
        return weighted_loss.mean()


def plot_results(original_rnn,r,epoch,sv=1,T=0,idx=0,post = 0):
    a = original_rnn.m.detach().cpu().numpy()
    c = original_rnn.n.detach().cpu().numpy()
    b = original_rnn.b.detach().cpu().numpy()

    kappas = c@r
    
    
    w = original_rnn.W_out.detach().cpu().numpy()
    b_out = original_rnn.b_out.detach().cpu().numpy()



    x_min = np.min(kappas[0,:])-10
    x_max = np.max(kappas[0,:])+10
    y_min = np.min(kappas[1,:])-10
    y_max = np.max(kappas[1,:])+10
    
    
    
    def dkappa_func(x,y,a,c,b):
        z1 = np.zeros(x.shape)
        z2 = np.zeros(x.shape)
        for i in range(a.shape[0]):
            z1 = z1 +  c[0,i] * np.tanh(a[i,0] * x + a[i,1] * y + b[i])
            z2 = z2 +  c[1,i] * np.tanh(a[i,0] * x + a[i,1] * y+b[i])        
        return (z1-x,z2-y)


    x = np.linspace(x_min,x_max,15);
    y = np.linspace(y_min,y_max,15);
    

    X, Y = np.meshgrid(x,y)
    [z1,z2] = dkappa_func(X,Y,a,c,b)

    #N = np.sqrt(z1**2+z2**2)  # there may be a faster numpy "normalize" function
    #z1, z2 = z1/N, z2/N

    plt.quiver(X,Y,z1,z2,scale = 100,width = .002,headwidth = 5)
    plt.scatter(kappas[0,:total_time],kappas[1,:total_time],10)
    plt.scatter(kappas[0,total_time:],kappas[1,total_time:],5,marker = '.')
    plt.scatter(kappas[0,0],kappas[1,0],marker = 'x')
    plt.scatter(kappas[0,T+1:T+2],kappas[1,T+1:T+2],marker = 'o')
    plt.scatter(kappas[0,total_time],kappas[1,total_time],marker = '>')
    plt.xlim([x_min,x_max])
    plt.ylim([y_min,y_max])
    
    
    if w[1] != 0:  # Ensure w[1] is not zero to avoid division by zero
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = -(w[0] / w[1]) * x_vals - b_out / w[1]
        y_vals_upper = y_vals + 1.1/w[1]
        y_vals_lower = y_vals - 1.1/w[1]
        plt.plot(x_vals, y_vals, 'r--', label='Decision Boundary')  # Dashed red line
        plt.fill_between(x_vals, y_vals, y_vals_upper, color='green', alpha=0.2, label='Margin')
        plt.fill_between(x_vals, y_vals_lower, y_vals, color='red', alpha=0.2, label='Margin')


    if sv == 1:
        plt.savefig(f'./results/post_{post}/delay_{T}/process_{idx}/epoch_{epoch}.pdf')
        plt.close()

class RankKRNN(nn.Module):
    def __init__(self, N, alpha,K=2):
        super(RankKRNN, self).__init__()
        self.N = N
        self.alpha = alpha
        self.m = nn.Parameter(torch.randn(N,K) / np.sqrt(N))
        self.n = nn.Parameter(torch.randn(K,N) / np.sqrt(N))
        self.b = nn.Parameter(torch.zeros(N))
        self.W_out = nn.Parameter(torch.randn(2) / np.sqrt(2))
        self.b_out = nn.Parameter(torch.randn(1) / np.sqrt(1))
        self.noise_std = 0.0

    def forward(self, x):
        phi = torch.tanh(self.m @self.n @x + self.b)
        x = (1 - self.alpha) * x + self.alpha * phi
        out = torch.sigmoid(self.W_out  @ self.n @x + self.b_out)
        return x,out



def generate_target(T_disc, total_time,post):
    if post == 1:
        return torch.cat([torch.zeros(T_disc), torch.ones(int(0.25*(total_time - T_disc))),
                     torch.zeros(int(0.75*(total_time - T_disc)))])
    else:
        return torch.cat([torch.zeros(T_disc), torch.ones(int(0.25*(total_time - T_disc)))])


def train_rnn(T, total_time, learning_rate, num_epochs,post,idx):
    T_disc = int(T)
    seed = idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    N = 100
    alpha = 0.5
    K = 2
    rnn = RankKRNN(N, alpha,K)


    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
    target = generate_target(T_disc, total_time,post)

    os.makedirs(f'./results/post_{post}/delay_{T}/process_{idx}', exist_ok=True)
    
    if post == 0:
        offset = 6
    else:
        offset = 0
    
    loss_history = []
    output_history = []
    grad_history = []
    
    pbar = tqdm(range(num_epochs))
    x = torch.randn(rnn.N)/10

    for epoch in pbar:
        optimizer.zero_grad()
        output = []
        x_all = np.zeros([ total_time+1,100])
        
        x_epoch = x.clone()
        x_all[0,:] = x_epoch.cpu().detach().numpy()

        for t in range(total_time-offset):
            x_epoch, out = rnn(x_epoch)
            output.append(out)
            x_all[t+1,:] = x_epoch.cpu().detach().numpy()

        output = torch.stack(output).squeeze()

        loss = torch.sum((output - target)**2) / T
        
        weights = torch.zeros_like(target).to('cpu')
        

        wd = T/target.shape[0]
        weights[:T] = 1-wd
        weights[T:] = wd

        criterion = WeightedMSELoss(weights)
        
        loss = criterion(output,target)
        
        loss.backward()
        
        optimizer.step()

        loss_history.append(loss.item())
        output_history.append(output.detach().numpy())
        # Step 6: Compute the L2 norm of the gradient
        l2_norm = 0.0
        for param in rnn.parameters():
            if param.grad is not None:
                l2_norm += param.grad.norm(2).item() ** 2
        
        l2_norm = l2_norm ** 0.5  # Take the square root to get the L2 norm
        grad_history.append(l2_norm)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        
        #if epoch % 1000 == 0:
        #    print(f"{rnn_type} RNN - Epoch {epoch}, Loss: {loss.item()}")
        if epoch % 1000 == 0:
            
                
            # For arrange the firing rates
            def get_ranked_indices(r,thr = 0.5):
                n_cell = r.shape[1]
                max_active = np.zeros(n_cell)
                for i in range(n_cell):
                    temp = r[:,i]
                    inds = np.argsort(temp)[::-1]
                    if max(temp) < thr:
                        max_active[i] = 99999999
                    else:
                        max_active[i] = inds[0]
                    
                inds_all = np.argsort(max_active)
                return inds_all
                
            x_new = np.zeros([N,3*total_time+1])
            x_epoch = x.clone()
            x_new[:,0]= x.clone().detach().cpu().numpy()
            for t in range(3*total_time):
                with torch.no_grad():
                    x_epoch, _ = rnn(x_epoch)
                    x_new[:,t+1]= x_epoch.detach().cpu().numpy()
                    
            
            
            plot_results(rnn,np.array(x_new),epoch,1,T,idx,post)
    np.savez(f'./results/post_{post}/delay_{T}/process_{idx}/results.npz',loss_history = loss_history,grad_history = grad_history)


# Main code 
import random
learning_rate = .005

for t in range(8):
    T = t*2+6
    total_time = T +8
    if total_time < 18:
        num_epochs = 500000
    else:
        num_epochs = 1500000
    for post in range(2):
        Parallel(n_jobs=10)(delayed(train_rnn)(T, total_time, learning_rate, num_epochs,post,i) for i in range(100))
        




