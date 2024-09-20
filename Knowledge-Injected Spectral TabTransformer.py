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
import random
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import TabTransformer
from sklearn.preprocessing import LabelEncoder
import torchviz


# 加载数据
data_path = r'F:\game\小论文\paper1\数据\21个材料\传统方法获取平均光谱.xlsx'
label_path = r'F:\game\小论文\paper1\数据\21个材料\材料名与油分含量已聚类剔除按平均光谱排序21材料.xlsx'
data_df = pd.read_excel(data_path, sheet_name='Sheet1', header=None)
data1 = data_df.iloc[1:330, 0:257]
data1 = data1.apply(pd.to_numeric, errors='coerce')
data1 = data1.fillna(data1.mean())
data1 = np.round(data1, decimals=4)
data = np.array(data1)

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


model = TabTransformer(
    categories=[22],      # tuple containing the number of unique values within each category
    num_continuous=256,                # number of continuous values
    dim=32,                           # dimension, paper set at 32
    dim_out=1,                        # binary prediction, but could be anything
    depth=6,                          # depth, paper recommended 6
    heads=4,                          # heads, paper recommends 8
    attn_dropout=0.1,                 # post-attention dropout
    ff_dropout=0.1,                   # feed forward dropout
    mlp_hidden_mults=(4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act=nn.ReLU()               # activation for final mlp, defaults to relu, but could be anything else (selu etc)
)

# Assume you have the following dummy input for categorical and continuous data
x_categ_dummy = torch.randint(0, 22, (1, 1)).long()  # One categorical feature
x_cont_dummy = torch.randn(1, 256)  # 256 continuous features

# Visualize the model
y_dummy = model(x_categ_dummy, x_cont_dummy)
dot = torchviz.make_dot(y_dummy, params=dict(list(model.named_parameters()) + [('x_categ', x_categ_dummy), ('x_cont', x_cont_dummy)]))
dot.render('TabTransformer_model_visualization', format='png')


# Prepare categorical features
label_encoder = LabelEncoder()
all_categ_features = np.concatenate([data[:, 0], data[:, 0]])
label_encoder.fit(all_categ_features)
x_categ_train = label_encoder.transform(X_train[:, 0])
x_categ_test = label_encoder.transform(X_test[:, 0])
x_categ_train = torch.tensor(x_categ_train).long().unsqueeze(1)
x_categ_test = torch.tensor(x_categ_test).long().unsqueeze(1)

# Prepare continuous features
x_cont_train = torch.tensor(X_train[:, 1:]).float()
x_cont_test = torch.tensor(X_test[:, 1:]).float()

# Convert labels to tensors
y_train_tensor = torch.tensor(y_train).float().unsqueeze(1)
y_test_tensor = torch.tensor(y_test).float().unsqueeze(1)

num_epochs = 100
learning_rate = 0.000003
weight_decay = 0.001
batch_size = 8

# Data loader
train_dataset = TensorDataset(x_categ_train, x_cont_train, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# output = model(x_categ_train, x_cont)  # 传入模型的调用方式
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training and evaluation
train_losses = []
test_losses = []
train_rmses = []
test_rmses = []
train_r2s = []
test_r2s = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for x_categ, x_cont, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x_categ, x_cont)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    train_losses.append(total_train_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        y_pred_train = model(x_categ_train, x_cont_train)
        train_rmse = np.sqrt(mean_squared_error(y_train_tensor.numpy(), y_pred_train.numpy()))
        train_rmses.append(train_rmse)

        train_r2 = r2_score(y_train_tensor.numpy(), y_pred_train.numpy())
        train_r2s.append(train_r2)

        y_pred_test = model(x_categ_test, x_cont_test)
        test_loss = criterion(y_pred_test, y_test_tensor).item()
        test_losses.append(test_loss)

        test_rmse = np.sqrt(mean_squared_error(y_test_tensor.numpy(), y_pred_test.numpy()))
        test_rmses.append(test_rmse)
        test_r2 = r2_score(y_test_tensor.numpy(), y_pred_test.numpy())
        test_r2s.append(test_r2)
    print(f'Epoch {epoch + 1}, Train Loss: {total_train_loss / len(train_loader):.4f}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}, Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}')

# 保存模型
torch.save(model.state_dict(), 'Tab-transformer.pth')
print("模型已保存至文件 TabTransformer_spectral_model.pth")

metrics = pd.DataFrame({
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Train RMSE': train_rmses,
    'Test RMSE': test_rmses,
    'Train R2': train_r2s,
    'Test R2': test_r2s
})

# 将 DataFrame 保存到 Excel 文件中
metrics.to_excel('F:\game\小论文\paper1\结果\深度学习\\21材料\\transformer-实验\\Tab-transformer.xlsx', index=False)
print("性能指标已保存至文件 model_performance_metrics.xlsx")

# Visualization of training and testing losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss per Epoch')
plt.legend()
plt.show()



