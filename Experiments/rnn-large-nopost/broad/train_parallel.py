import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the training function
def train_rnn_model(process_idx,delay_interval,num_epochs,learning_rate,sh=0,sv=1):
    
    
    import src.data as data
    import src.rnn as rnn
    import torch.optim as optim
    import torch.nn as nn
    
    
    def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
    
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

    # Set the seed based on the process ID
    seed = process_idx
    seed_everything(seed)

    # Hyperparameters
    
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

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-7)
    

    # Metrics Tracking
    loss_list = []
    reaction_accuracy_list = []
    reaction_reliability_list = []
    W_rec_grad_list = []

    # Training Loop
    epoch_bar = range(num_epochs)
    for epoch in epoch_bar:
        for _, (inp, gt_out) in enumerate(dataloader):
            inp, gt_out = inp.to(device), gt_out.to(device)
            
            
            
            weight = 10/gt_out.shape[1]
            weights = torch.zeros_like(gt_out).to(device)
            
            weights[:,-10:,:] = 1-weight;
            weights[:,:-10,:] = weight;
         
            
            
            
            # Loss Function
            criterion = WeightedMSELoss(weights)

            #criterion = nn.MSELoss()

            optimizer.zero_grad()

            pred_out_stack, _ = model(inp)

            loss = criterion(pred_out_stack, gt_out)
            reaction_accuracy, reaction_reliability = model.calculate_accuracy(
                pred_out_stack, gt_out,
                reaction_start=int(((input_interval+delay_interval)/delta_t)-1), 
                reaction_end=int(((input_interval+delay_interval+reaction_interval)/delta_t)-1)
            )

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            reaction_accuracy_list.append(reaction_accuracy)
            reaction_reliability_list.append(reaction_reliability)
            W_rec_grad_list.append(torch.norm(model.W_rec.weight.grad).cpu().item())

            
        # Save model and plots periodically
        if (epoch) % 50 == 0:  # Adjust save frequency as needed
            if sv:
                os.makedirs(f'./checkpoints/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}', exist_ok=True)
                torch.save(model.state_dict(), f'./checkpoints/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/model_epoch_{epoch + 1}.pt')
                np.savez(f'./checkpoints/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/training_details.npz',
                         loss_list=loss_list,
                         reaction_accuracy_list=reaction_accuracy_list,
                         reaction_reliability_list=reaction_reliability_list,
                         W_rec_grad_list=W_rec_grad_list)
            # Save plots
            fig, axs = plt.subplots(2, 4, figsize=(15, 7.5))
    
            # Row 1
            axs[0, 0].set_title('Loss')
            axs[0, 0].plot(torch.arange(len(loss_list)), loss_list)
            axs[0, 0].set_ylim(-0.01, max(loss_list)+0.01)
            
            axs[0, 1].set_title('W_rec Gradient Norm')
            axs[0, 1].plot(torch.arange(len(W_rec_grad_list)), W_rec_grad_list)
            axs[0, 1].set_ylim(-0.01, max(W_rec_grad_list)+0.01)
    
            axs[0, 2].set_title('Reaction Accuracy')
            axs[0, 2].plot(torch.arange(len(reaction_accuracy_list)), reaction_accuracy_list)
            axs[0, 2].set_ylim(-0.05, 1.05)
    
            axs[0, 3].set_title('Reaction Reliability')
            axs[0, 3].plot(torch.arange(len(reaction_reliability_list)), reaction_reliability_list)
            axs[0, 3].set_ylim(-0.05, 1.05)
    
            # Row 2
            rand_idx = torch.randint(len(dcdt), size=[1])[0].item()
            axs[1, 0].set_title('Example Input')
            axs[1, 0].plot(torch.arange(inp.to('cpu').shape[1]), inp[rand_idx, :, 0].to('cpu'))
            axs[1, 0].plot(torch.arange(inp.shape[1]).to('cpu'), inp[rand_idx, :, 1].to('cpu'))
    
            axs[1, 1].set_title('GT Output')
            axs[1, 1].plot(torch.arange(gt_out.to('cpu').shape[1]), gt_out.to('cpu')[rand_idx, :, 0], label='GT Ch1',color = 'blue')
            axs[1, 1].plot(torch.arange(gt_out.to('cpu').shape[1]), gt_out.to('cpu')[rand_idx, :, 1], label='GT Ch2',color = 'red')
            axs[1, 1].plot(torch.arange(pred_out_stack.to('cpu').shape[1]), pred_out_stack.to('cpu')[rand_idx, :, 0].detach().cpu(), label='GT Ch1',color = 'blue',ls = '--')
            axs[1, 1].plot(torch.arange(pred_out_stack.to('cpu').shape[1]), pred_out_stack.to('cpu')[rand_idx, :, 1].detach().cpu(), label='GT Ch2',color = 'red',ls = '--')
            
            
            extended_duration = 5 * dcdt.dataset.shape[2]
            extended_inputs = torch.zeros((dcdt.dataset.shape[0],2, extended_duration, dcdt.dataset.shape[-1])).to(device)
            # Correctly assign the original data to the extended input
            extended_inputs[:, :,:dcdt.dataset.shape[2], :] = dcdt.dataset[:, :,:, :].to(device)

            with torch.no_grad():
                output_extended, firing_rate_stack_extended = model(extended_inputs[:,0,:,:])
            
            firing_rates_new = firing_rate_stack_extended[0].detach().cpu().numpy()
            sorted_indices = get_ranked_indices(abs(firing_rates_new))
            
            axs[1, 2].set_title('Pred Output')
            axs[1, 2].plot(torch.arange(extended_inputs[:,1,:,:].to('cpu').shape[1]), extended_inputs[:,1,:,:].to('cpu')[rand_idx, :, 0], label='GT Ch1',color = 'blue')
            axs[1, 2].plot(torch.arange(extended_inputs[:,1,:,:].to('cpu').shape[1]), extended_inputs[:,1,:,:].to('cpu')[rand_idx, :, 1], label='GT Ch2',color = 'red')

            axs[1, 2].plot(torch.arange(output_extended.to('cpu').shape[1]), output_extended.to('cpu')[rand_idx, :, 0].detach().cpu(), label='Pred Ch1',color = 'blue',ls = '--')
            axs[1, 2].plot(torch.arange(output_extended.to('cpu').shape[1]), output_extended.to('cpu')[rand_idx, :, 1].detach().cpu(), label='Pred Ch2',color = 'red',ls = '--')
    
            axs[1, 2].set_ylim(-0.05, 1.05)
    
            axs[1, 3].set_title('Firing Rates Through Time')
    
            axs[1, 3].imshow(abs(firing_rates_new[:, sorted_indices]).T, interpolation='None', aspect='auto')
    
            # Save the plot
            plt.tight_layout()
            if sh==1:
                plt.show()
            if sv == 1:
                plt.savefig(f'./checkpoints/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/metrics_plot_epoch_cur.png')
            plt.close()

    # Save final model
    if sv == 1:
        os.makedirs(f'./results/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}', exist_ok=True)
        torch.save(model.state_dict(), f'./results/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/model_final.pt')
        np.savez(f'./results/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/training_details.npz',
                 loss_list=loss_list,
                 reaction_accuracy_list=reaction_accuracy_list,
                 reaction_reliability_list=reaction_reliability_list,
                 W_rec_grad_list=W_rec_grad_list)
    # Save plots
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.5))
    
    # Row 1
    axs[0, 0].set_title('Loss')
    axs[0, 0].plot(torch.arange(len(loss_list)), loss_list)
    axs[0, 0].set_ylim(-0.001, max(loss_list)+0.01)
    
    axs[0, 1].set_title('W_rec Gradient Norm')
    axs[0, 1].plot(torch.arange(len(W_rec_grad_list)), W_rec_grad_list)
    axs[0, 1].set_ylim(-0.001, max(W_rec_grad_list)+0.01)
    
    axs[0, 2].set_title('Reaction Accuracy')
    axs[0, 2].plot(torch.arange(len(reaction_accuracy_list)), reaction_accuracy_list)
    axs[0, 2].set_ylim(-0.05, 1.05)
    
    axs[0, 3].set_title('Reaction Reliability')
    axs[0, 3].plot(torch.arange(len(reaction_reliability_list)), reaction_reliability_list)
    axs[0, 3].set_ylim(-0.05, 1.05)
    
    # Row 2
    rand_idx = torch.randint(len(dcdt), size=[1])[0].item()
    axs[1, 0].set_title('Example Input')
    axs[1, 0].plot(torch.arange(inp.to('cpu').shape[1]), inp[rand_idx, :, 0].to('cpu'))
    axs[1, 0].plot(torch.arange(inp.shape[1]).to('cpu'), inp[rand_idx, :, 1].to('cpu'))
    
    axs[1, 1].set_title('GT Output')
    axs[1, 1].plot(torch.arange(gt_out.to('cpu').shape[1]), gt_out.to('cpu')[rand_idx, :, 0], label='GT Ch1',color = 'blue')
    axs[1, 1].plot(torch.arange(gt_out.to('cpu').shape[1]), gt_out.to('cpu')[rand_idx, :, 1], label='GT Ch2',color = 'red')
    axs[1, 1].plot(torch.arange(pred_out_stack.to('cpu').shape[1]), pred_out_stack.to('cpu')[rand_idx, :, 0].detach().numpy(), label='GT Ch1',color = 'blue',ls = '--')
    axs[1, 1].plot(torch.arange(pred_out_stack.to('cpu').shape[1]), pred_out_stack.to('cpu')[rand_idx, :, 1].detach().numpy(), label='GT Ch2',color = 'red',ls = '--')
    
    
    extended_duration = 5 * dcdt.dataset.shape[2]
    extended_inputs = torch.zeros((dcdt.dataset.shape[0],2, extended_duration, dcdt.dataset.shape[-1])).to(device)
    # Correctly assign the original data to the extended input
    extended_inputs[:, :,:dcdt.dataset.shape[2], :] = dcdt.dataset[:, :,:, :].to(device)
    
    with torch.no_grad():
        output_extended, firing_rate_stack_extended = model(extended_inputs[:,0,:,:])
    
    firing_rates_new = firing_rate_stack_extended[0].detach().cpu().numpy()
    sorted_indices = get_ranked_indices(abs(firing_rates_new))
    
    axs[1, 2].set_title('Pred Output')
    axs[1, 2].plot(torch.arange(output_extended.to('cpu').shape[1]), output_extended.to('cpu')[rand_idx, :, 0].detach().cpu(), label='Pred Ch1')
    axs[1, 2].plot(torch.arange(output_extended.to('cpu').shape[1]), output_extended.to('cpu')[rand_idx, :, 1].detach().cpu(), label='Pred Ch2')
    
    axs[1, 2].set_ylim(-0.05, 1.05)
    
    axs[1, 3].set_title('Firing Rates Through Time')
    
    axs[1, 3].imshow(abs(firing_rates_new[:, sorted_indices]).T, interpolation='None', aspect='auto')
    
    # Save the plot
    plt.tight_layout()
    if sv == 1:
        plt.savefig(f'./results/lr{learning_rate}_delay_{delay_interval}/process_{process_idx}/metrics_plot_final.png')
    plt.close()

    return f"Process {process_idx} completed."


lr_all = np.logspace(-1,-3,15);

# Number of processes to run
n_processes = 20
n_exp = 100
for k in range(15):
    learning_rate = lr_all[k]
    num_epochs = int( np.maximum(1e3,3e4 *1e-3 / learning_rate))
    for l in reversed(range(21)):
        delay_interval = l*0.020
        results = Parallel(n_jobs=n_processes)(delayed(train_rnn_model)(i,delay_interval,num_epochs,learning_rate) for i in range(n_exp))
        print(f'Learning rate {learning_rate}. delay {delay_interval}. done.\n')