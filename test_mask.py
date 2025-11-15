from utils.masking import generate_block_mask_train, generate_block_mask_sample
import torch

seq_len = 6
block_size = 2
n = 3

mask = generate_block_mask_sample(
                b=None, h=None, q_idx=torch.arange(seq_len)[:, None], 
                kv_idx=torch.arange(seq_len)[None, :], block_size=block_size, n=n)
print(mask)