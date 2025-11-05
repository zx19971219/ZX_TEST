from utils.masking import generate_partial_mask, generate_causal_mask, generate_self_only_mask, generate_block_mask_encoder, generate_block_mask_src, generate_block_mask_tgt
import torch

seq_len = 8
block_size = 2

mask = generate_block_mask_src(
                b=None, h=None, q_idx=torch.arange(seq_len)[:, None], 
                kv_idx=torch.arange(seq_len)[None, :], block_size=block_size)
print(~mask)