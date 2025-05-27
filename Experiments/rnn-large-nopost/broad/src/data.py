#%%
import torch
from torch.utils.data import Dataset

#%%
class DCDT(Dataset):
    # Delayed Cue Discrimination Task
    def __init__(self, 
                 num_of_samples_per_each_class=1024,
                 input_interval=0.5, 
                 delay_interval=1.0, 
                 reaction_interval=1.0, 
                 post_reaction_interval=0.5,
                 delta_t=50e-3):
        super().__init__()

        self.batch_size = num_of_samples_per_each_class

        self.input_interval = input_interval
        self.delay_interval = delay_interval
        self.reaction_interval = reaction_interval
        self.post_reaction_interval = post_reaction_interval

        self.delta_t = delta_t

        self.dataset = self.create_dataset() # B, [in, out], T, 2

    def ground_truths(self):
        return self.create_dataset(1)
    
    def get_num_of_samples(self):
        return {
            "num_in_samples": int(self.input_interval / self.delta_t),
            "num_delay_samples": int(self.delay_interval / self.delta_t),
            "num_reaction_samples": int(self.reaction_interval / self.delta_t),
            "num_post_reaction_samples": int(self.post_reaction_interval / self.delta_t)
        }

    def create_dataset(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        sample_labels = torch.arange(2).view(1, 2)
        sample_labels = torch.cat((sample_labels, torch.abs(sample_labels-1)), dim=0) # [[0,1], [1,0]]

        num_of_samples = self.get_num_of_samples()
        
        inputs = sample_labels.unsqueeze(1).repeat(1, num_of_samples['num_in_samples'], 1) # [2, num_in_samples, 2]
        reaction = sample_labels.unsqueeze(1).repeat(1, num_of_samples['num_reaction_samples'], 1) # [2, num_reaction_samples, 2]

        if num_of_samples['num_delay_samples'] != 0:
            delays = torch.zeros(2, num_of_samples['num_delay_samples'], 2)

        if num_of_samples['num_post_reaction_samples'] != 0:
            post_reactions = torch.zeros(2, num_of_samples['num_post_reaction_samples'], 2)

        
        dataset = inputs
        if num_of_samples['num_delay_samples'] != 0:
            dataset = torch.cat((dataset, delays), dim=1)
        dataset = torch.cat((dataset, torch.zeros(2, num_of_samples['num_reaction_samples'], 2)), dim=1)
        if num_of_samples['num_post_reaction_samples'] != 0:
            dataset = torch.cat((dataset, post_reactions), dim=1)
        
        dataset.unsqueeze_(1) # 2, 1, T, 2

        dataset = torch.cat((dataset, torch.cat((torch.zeros(2, 1, num_of_samples['num_in_samples']+num_of_samples['num_delay_samples'], 2), reaction.unsqueeze(1), torch.zeros(2, 1, num_of_samples['num_post_reaction_samples'], 2)), dim=2)), dim=1) # 2, 2, T, 2

        return dataset.repeat(batch_size, 1, 1, 1) # B*2, 2, T, 2

    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1] # input, out