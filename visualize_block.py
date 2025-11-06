import numpy as np
import matplotlib.pyplot as plt

# 1. 读取保存的数据
data = np.load('./outputs/test_results/ETTh2/results.npz')
preds = data['preds']
trues = data['trues']

print(f"preds shape: {preds.shape}, trues shape: {trues.shape}")

# 2. 选择要可视化的样本和特征
sample_id = 0      # 第一个样本
feature_id = 0     # 第一个特征

y_pred = preds[sample_id, :, feature_id]
y_true = trues[sample_id, :, feature_id]

# 3. 绘图
plt.figure(figsize=(10, 5))
plt.plot(y_true, label='Ground Truth', color='black')
plt.plot(y_pred, label='Prediction', color='red', linestyle='--')

plt.title(f'Sample {sample_id}, Feature {feature_id}: Prediction vs Ground Truth')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
