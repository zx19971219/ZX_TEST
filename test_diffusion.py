import numpy as np
import torch
import matplotlib.pyplot as plt
from layers.TimeDART_EncDec import Diffusion

# 读取数据
data = np.load('./outputs/test_results/ETTh2/results.npz')
trues = data['trues']

# 将numpy array转换为torch tensor（保留原始数据类型，默认在CPU上）
# 如果需要在GPU上运行，可以添加 .to(device)，后续会统一处理
trues = torch.tensor(trues, dtype=torch.float32)  # 建议用float32适配深度学习模型

time_steps = 20
device = 'cuda:0'
scheduler = 'cosine'

diffusion = Diffusion(
    time_steps=time_steps,
    device=device,
    scheduler=scheduler,
    block_size=4,
)

# 将trues移到指定设备（CPU/GPU），保持和diffusion模型同设备
trues = trues.to(device)

sample_id = 0
feature_id = 0
# 从torch tensor中索引数据，结果仍是torch tensor
y_true = trues[sample_id, :, feature_id]

# 确保输入diffusion的是torch tensor（添加unsqueeze增加batch和channel维度，多数扩散模型需要）
# 从 (seq_len,) -> (1, seq_len, 1) 或 (1, 1, seq_len)，根据Diffusion模型要求调整
# 这里假设模型需要 (batch_size, seq_len, feature_dim) 格式
y_true_reshaped = y_true.unsqueeze(-1)  # 变成 (seq_len, 1)

# 生成带噪声的序列
noisy_series, _ = diffusion.noise(y_true_reshaped, t=time_steps-1)

gaussian_noise = np.random.randn(*y_true_reshaped.shape)

# 还原维度用于绘图（从tensor转为numpy array）
y_true_np = y_true.cpu().numpy()
noisy_series_np = noisy_series.squeeze().cpu().numpy()  # 去除多余维度

# 绘图部分保持不变
plt.figure(figsize=(10,5))
plt.plot(y_true_np, label='Ground Truth', color='black')
plt.plot(noisy_series_np, label='Noisy Series', color='red', linestyle='--')
plt.plot(guassian_noise, label='Guassian Noise', color='blue', linestyle='-.')

plt.title(f'Sample {sample_id}, Feature {feature_id}')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("diffusion_plot.png")  # 保存图像
plt.show()                          # 显示图像