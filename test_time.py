import torch 

ddim_method = 'quad'
time_steps = 1000
sample_steps = 50

# if ddim_method == 'uniform':
#     c = time_steps // sample_steps
#     timesteps = torch.arange(0, time_steps, c, dtype=torch.long)
            
# elif ddim_method == 'quad':
#     t = torch.linspace(0, (time_steps * 0.8) ** 0.5, sample_steps)
#     timesteps = (t ** 2).long()
        
# timesteps = timesteps + 1
# timesteps_prev = torch.cat([torch.zeros(1, dtype=torch.long), timesteps[:-1]])  
# timesteps = timesteps.flip(0)
# timesteps_prev = timesteps_prev.flip(0)

timesteps = torch.linspace(0, time_steps - 1, sample_steps, dtype=torch.long)
timesteps_prev = torch.cat([torch.tensor([-1], dtype=torch.long), timesteps[:-1]])
    
print(timesteps)
print(timesteps_prev)