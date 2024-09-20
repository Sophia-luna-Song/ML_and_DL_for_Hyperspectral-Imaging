import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from OpenSA.DataLoad.DataLoad import ks
from sklearn.model_selection import train_test_split
import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU也需要设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(46)
# 46 目前效果最好

# 定义模型
class OilContentPredictor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(OilContentPredictor, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加序列长度维度
        embedded = self.embedding(x)
        transformed = self.transformer(embedded)
        transformed = self.dropout(transformed)
        output = self.fc_out(transformed.mean(dim=1))

        return output


data_path = r'F:\game\小论文\paper1\数据\21个材料\传统方法获取平均光谱.xlsx'
label_path = r'F:\game\小论文\paper1\数据\21个材料\材料名与油分含量已聚类剔除按平均光谱排序21材料.xlsx'
# data_path = r'F:\game\小论文\paper4-花生\数据\38品种数据\merged_output_已排序.xlsx'
# data_path = r'F:\game\小论文\paper1\数据\传统方法获取平均光谱.xlsx'
data_df = pd.read_excel(data_path, sheet_name='Sheet1', header=None)
data1 = data_df.iloc[1:330, 1:257]
data1 = data1.apply(pd.to_numeric, errors='coerce')
data1 = data1.fillna(data1.mean())
data1 = np.round(data1, decimals=4)
data = np.array(data1)

# 加载标签
# label_path = r'F:\game\小论文\paper4-花生\数据\38品种数据\20240113234428.xlsx'
label_df = pd.read_excel(label_path)
label = label_df.iloc[0:330, 3].values
# 确保所有值都是浮点数
label = label.astype(float)
label = np.round(label, decimals=2)
label = copy.deepcopy(label)

# X_train, X_test, y_train, y_test = ks(data, label, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 创建TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# # 准备数据
# spectra_tensor = torch.tensor(data, dtype=torch.float32)
# oil_content_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(1)
#
# # 数据切分
# dataset = TensorDataset(spectra_tensor, oil_content_tensor)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# pretrained_params = torch.load("I:\PythonProject\OpenSA-main\OpenSA\深度学习组分预测\自监督学习\\transformer_spectral_model.pth")
# 初始化模型、损失函数和优化器
model = OilContentPredictor(input_dim=256, model_dim=256, num_heads=4, num_layers=3, output_dim=1)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000003, weight_decay=1e-3)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimi.ozer, T_max=100)  # T_max是周期
# optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 打印网络模型结构
def print_layer_info(model, input_tensor):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            print(f"Layer: {name} - {layer}")
            input_tensor = layer(input_tensor)
            print(f"Output Shape: {input_tensor.shape}\n")
        elif isinstance(layer, nn.TransformerEncoder):
            print(f"Layer: {name} - {layer}")
            input_tensor = layer(input_tensor)
            print(f"Output Shape: {input_tensor.shape}\n")
        elif isinstance(layer, nn.TransformerEncoderLayer):
            print(f"Layer: {name} - {layer}")
            input_tensor = layer(input_tensor)
            print(f"Output Shape: {input_tensor.shape}\n")
        elif isinstance(layer, nn.Dropout):
            input_tensor = layer(input_tensor)
        elif isinstance(layer, nn.Sequential):  # If it's a sequential block, recurse into it
            print_layer_info(layer, input_tensor)
        else:
            print(f"Layer: {name} - {layer}")
            input_tensor = layer(input_tensor)
            print(f"Output Shape: {input_tensor.shape}\n")


# # 创建一个示例输入
# input_tensor = torch.randn(1, 256)
# print_layer_info(model, input_tensor)

num_epochs = 1800
train_losses = []
test_losses = []
train_rmses = []
train_r2s = []
test_rmses = []
test_r2s = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)


    model.eval()
    total_test_loss = 0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            test_preds.extend(outputs.numpy())
            test_targets.extend(labels.numpy())
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds)
    test_rmses.append(test_rmse)
    test_r2s.append(test_r2)

    total_train_loss = 0
    train_preds, train_targets = [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()
            train_preds.extend(outputs.numpy())
            train_targets.extend(labels.numpy())
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
    train_r2 = r2_score(train_targets, train_preds)
    train_rmses.append(train_rmse)
    train_r2s.append(train_r2)

    print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}, Test Loss: {avg_test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}')

torch.save(model.state_dict(), 'F:\game\小论文\paper1\结果\深度学习\\21材料\\transformer-实验\OilContentPredictor_model.pth')
print("模型已保存至文件 OilContentPredictor_model.pth")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# 创建一个DataFrame来存储训练和测试损失
losses_df = pd.DataFrame({
    'Train Losses': train_losses,
    'Train rmse': train_rmse,
    'Train r2': train_r2,
    'Test Losses': test_losses,
    'Test rmse': test_rmse,
    'Test r2': test_r2
})

# 保存DataFrame到Excel文件
losses_df.to_excel('F:\game\小论文\paper1\结果\深度学习\\21材料\\transformer-实验\\losses.xlsx', index=False)
print("损失数据已保存至 losses.xlsx")

