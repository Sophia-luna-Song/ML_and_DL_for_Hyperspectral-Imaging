import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random


class SpectralCNN(nn.Module):
    def __init__(self):
        super(SpectralCNN, self).__init__()
        # 卷积层和最大池化层定义
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=5, stride=2, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv1d(16, 8, kernel_size=5, stride=2, padding=1)

        # 注意力机制组件
        self.gap = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，输出大小为1
        self.att_fc1 = nn.Linear(8, 8)  # 注意力的全连接层，输出尺寸与输入尺寸相同以进行元素乘法

        # 最终输出层
        self.fc_final = nn.Linear(8, 1)  # 最终输出层

    def forward(self, x):
        # 卷积层和池化层传递
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))

        # 注意力机制
        gap = self.gap(x)  # 全局平均池化
        gap = gap.view(gap.size(0), -1)  # 这个展平操作只是为了匹配全连接层的期望输入形式
        attention = torch.sigmoid(self.att_fc1(gap))  # 计算注意力权重
        x = x * attention.view(x.size(0), -1, 1)  # 重塑注意力权重以匹配x的维度并进行元素乘法

        # 展平操作
        x = x.view(x.size(0), -1)

        # 最终全连接输出层
        x = self.fc_final(x)
        return x


data_path = r'F:\game\小论文\paper1\数据\21个材料\传统方法获取平均光谱.xlsx'
label_path = r'F:\game\小论文\paper1\数据\21个材料\材料名与油分含量已聚类剔除按平均光谱排序21材料.xlsx'

data_df = pd.read_excel(data_path, sheet_name='Sheet1', header=None)
data1 = data_df.iloc[1:346, 1:257]
data1 = data1.apply(pd.to_numeric, errors='coerce')
data1 = data1.fillna(data1.mean())
data1 = np.round(data1, decimals=4)
data = np.array(data1)

# 加载标签
label_df = pd.read_excel(label_path)
label = label_df.iloc[0:346, 3].values
label = np.round(label, decimals=2)

# X_train, X_test, y_train, y_test = ks(data, label, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

# 检查数据类型
print(data_df.dtypes)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 创建TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)


# 创建模型实例
model = SpectralCNN()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003, betas=(0.99, 0.999), weight_decay=3e-2)


# 函数计算RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# 初始化记录结果的字典
results = {
    "Epoch": [],
    "Training Loss": [],
    "Training RMSE": [],
    "Training R2": [],
    "Test Loss": [],  # 新增测试损失记录
    "Test RMSE": [],
    "Test R2": []
}

# 训练模型
for epoch in range(1000):
    model.train()
    train_losses = []
    train_targets = []
    train_outputs = []

    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_targets.extend(labels.numpy())
        train_outputs.extend(outputs.detach().numpy())

    train_loss = np.mean(train_losses)
    train_rmse = rmse(np.array(train_outputs), np.array(train_targets))
    train_r2 = r2_score(np.array(train_targets), np.array(train_outputs))

    print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")

    # 测试模型
    model.eval()
    test_losses = []
    test_targets = []
    test_outputs = []
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        test_loss = criterion(outputs, labels)  # 计算测试损失
        test_losses.append(test_loss.item())
        test_targets.extend(labels.numpy())
        test_outputs.extend(outputs.detach().numpy())

    test_loss = np.mean(test_losses)  # 计算平均测试损失
    test_rmse = rmse(np.array(test_outputs), np.array(test_targets))
    test_r2 = r2_score(np.array(test_targets), np.array(test_outputs))
    print(f"Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")

    # 保存结果
    results["Epoch"].append(epoch + 1)
    results["Training Loss"].append(train_loss)
    results["Training RMSE"].append(train_rmse)
    results["Training R2"].append(train_r2)
    results["Test Loss"].append(test_loss)  # 保存测试损失
    results["Test RMSE"].append(test_rmse)
    results["Test R2"].append(test_r2)

# 保存结果到Excel
results_df = pd.DataFrame(results)
results_df.to_excel("training_results.xlsx", index=False)
