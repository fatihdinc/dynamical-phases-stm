#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self,
                 in_dim=2,
                 hid_dim=100,
                 out_dim=2,
                 alpha:float=0.1,
                 noise_sigma:float=0.001,
                 firing_rate_sigma:float=1.,
                 truncate_step:int=20):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.truncate_step = truncate_step

        self.noise_sigma = noise_sigma

        self.alpha = alpha
        self.firing_rate_sigma = firing_rate_sigma

        self.W_in = nn.Linear(in_dim, hid_dim, bias=True)
        self.W_rec = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W_out = nn.Linear(hid_dim, out_dim, bias=True)
        
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_rec.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

        with torch.no_grad():
            self.W_rec.weight.fill_diagonal_(0)
    
    def forward(self, inp):
        # inp --> B, T, in_dim
        time = inp.shape[1]
        firing_rates = F.tanh(torch.randn(inp.shape[0], self.hid_dim, requires_grad=True).to(inp.device) * self.firing_rate_sigma)

        pred_out_list = []
        firing_rate_list = []

        for t in range(time):
            if self.truncate_step != None and (t+1) % self.truncate_step == 0:
                firing_rates = firing_rates.detach()
            pred, firing_rates = self.forward_step(inp[:, t], firing_rates)

            pred_out_list.append(pred)
            firing_rate_list.append(firing_rates)
        
        return torch.stack(pred_out_list).permute(1, 0, 2), torch.stack(firing_rate_list).permute(1, 0, 2) # B, T, out_dim - B, T, hid_dim
    
    def forward_step(self, inp, firing_rate):
        # input --> B(Batch), D_in(R^{N_{in}})
        # fire_rate --> B(Batch), N+N_2

        # o^(t) = sigmoid(W_out @ r(t))
        pred_out = F.sigmoid(self.W_out(firing_rate)) # B, D_out

        # r[t + Δt] = (1 − α) * r[t] + α * tanh(W_rec @ r[t] + W_in @ I_in[t] + ε)
        hid = self.W_in(inp)
        rec = self.W_rec(firing_rate)
        noise = torch.randn(rec.shape).to(inp.device) * self.noise_sigma
        z = F.tanh(hid+rec+noise)

        pred_firing_rates = (1 - self.alpha) * firing_rate + self.alpha * z
        return pred_out, pred_firing_rates
    
    def calculate_accuracy(self, pred, gt, reaction_start, reaction_end):
        # x / sum(x) for obtaining pred probability distribution.
        pred_prob = torch.nan_to_num(pred / torch.sum(pred, dim=-1)[..., None])

        reaction_accuracy = 1 - F.l1_loss(torch.argmax(pred_prob[:, reaction_start:reaction_end, :], dim=-1).float(), torch.argmax(gt[:, reaction_start:reaction_end, :], dim=-1).float()) # B T_nos D_out -> 1 # Out should change correspond to input.
        reaction_reliability = torch.mean(torch.gather(pred_prob[:, reaction_start:reaction_end, :], -1, index=torch.argmax(gt[:, reaction_start:reaction_end, :], dim=-1)[..., None]))

        return torch.round(reaction_accuracy, decimals=4).item(), torch.round(reaction_reliability, decimals=4).item()