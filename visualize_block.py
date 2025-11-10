import numpy as np
import matplotlib.pyplot as plt
from layers.TimeDART_EncDec import Diffusion

# 读取数据
data = np.load('./outputs/test_results/ETTh2/results.npz')
preds = data['preds']
trues = data['trues']

sample_id = 0
feature_id = 0
y_pred = preds[sample_id, :, feature_id]
y_true = trues[sample_id, :, feature_id]

total_timesteps = len(y_pred)  # 预测序列的总时间步数
target_timestep = total_timesteps // 2  # 目标时间步（如总步长100则取50，步长99则取49）
target_pred_value = y_pred[target_timestep]  # 目标时间步对应的预测值
target_true_value = y_true[target_timestep]  # 目标时间步对应的真实值（可选显示）

plt.figure(figsize=(10,5))
plt.plot(y_true, label='Ground Truth', color='black')
plt.plot(y_pred, label='Prediction', color='red', linestyle='--')

plt.scatter(
    x=target_timestep,  # 目标时间步（x轴位置）
    y=target_pred_value,  # 对应预测值（y轴位置）
    color='blue',  # 标记颜色
    s=100,  # 标记大小
    marker='*',  # 标记形状（*为星号，还可设为'o'圆圈、's'正方形、'^'三角形）
    edgecolor='black',  # 标记边框颜色（增强对比度）
    label=f'Pred Marker (t={target_timestep})'  # 图例说明
)

# 可选：同时标记真实值（若需要对比）
plt.scatter(
    x=target_timestep,
    y=target_true_value,
    color='green',
    s=80,
    marker='s',
    edgecolor='black',
    label=f'True Marker (t={target_timestep})'
)

plt.title(f'Sample {sample_id}, Feature {feature_id}')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("prediction_plot.png")  # 保存图像
plt.show()                          # 显示图像
