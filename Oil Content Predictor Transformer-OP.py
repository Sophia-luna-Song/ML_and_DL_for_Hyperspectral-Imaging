import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import TabTransformer
from sklearn.preprocessing import LabelEncoder
from itertools import product


def loadDataset(batch_size, data_path, label_path):
    # 加载数据

    data_df = pd.read_excel(data_path, sheet_name='Sheet1', header=None)
    data1 = data_df.iloc[1:346, 1:257]
    data1 = data1.apply(pd.to_numeric, errors='coerce')
    data1 = data1.fillna(data1.mean())
    data1 = np.round(data1, decimals=4)
    data = np.array(data1)

    # 加载标签
    label_df = pd.read_excel(label_path)
    label = label_df.iloc[0:346, 1].values
    label = np.round(label, decimals=2)

    X_train, X_temp, y_train, y_temp = train_test_split(data, label, test_size=0.3, random_state=42)
    # 现在将临时集进一步划分为测试集和验证集，测试集占临时集的2/3（即全数据的20%），验证集占1/3（即全数据的10%）
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)

    # 转换数据为torch.Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


class OilContentPredictor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, activation='relu'):
        super(OilContentPredictor, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        # 使用传入的 activation_fn 参数指定激活函数
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, activation=activation)
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


# 网格搜索 超参数验证主体
def validT(model, optimizer, criterion, epochs, train_loader, val_loader):
    # Training and evaluation
    train_losses, train_rmses, train_r2s = [], [], []
    val_losses, val_rmses, val_r2s = [], [], []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)  # data 已经包括了形态和光谱特征
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss / len(train_loader))

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            y_pred_train = []
            y_true_train = []
            for data, targets in train_loader:
                outputs = model(data)  # 获取训练集预测结果
                y_pred_train.extend(outputs.numpy())  # 收集预测值
                y_true_train.extend(targets.numpy())  # 收集真实值
            train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))  # 计算训练集 RMSE
            train_r2 = r2_score(y_true_train, y_pred_train)  # 计算训练集 R2
            train_rmses.append(train_rmse)
            train_r2s.append(train_r2)

            # 对测试集进行评估
            y_pred_test = []
            y_true_test = []
            for data, targets in val_loader:
                outputs = model(data)  # 获取测试集预测结果
                y_pred_test.extend(outputs.numpy())  # 收集预测值
                y_true_test.extend(targets.numpy())  # 收集真实值
            val_loss = np.sqrt(mean_squared_error(y_true_test, y_pred_test))  # 计算测试集 RMSE
            val_losses.append(val_loss)  # 记录测试损失
            val_rmse = np.sqrt(val_loss)  # 计算测试 RMSE
            val_r2 = r2_score(y_true_test, y_pred_test)  # 计算测试 R2
            val_rmses.append(val_rmse)
            val_r2s.append(val_r2)

        # 输出每个epoch的训练和测试结果
        print(
            f'Epoch {epoch + 1}, Train Loss: {total_train_loss / len(train_loader):.4f}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}')

    return val_r2, val_rmse, val_loss


def evaluate_model(params, data_path, label_path):
    model = OilContentPredictor(input_dim=256, model_dim=params['dim'], num_heads=8, num_layers=params['depth'], output_dim=1, activation=params['activation'])
    num_epochs = 1600
    weight_decay = 0.001
    batch_size = params['batch_size']

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=weight_decay)

    train_loader, val_loader, test_loader = loadDataset(batch_size, data_path, label_path)
    val_r2, val_rmse, val_loss = validT(model, optimizer, criterion, num_epochs, train_loader, val_loader)

    # 返回评估指标
    return val_r2, val_rmse, val_loss


def best_model(model, optimizer, criterion, num_epochs, train_loader, valid_loader, test_loader):
    # Training and evaluation
    epochs = []
    train_losses, train_rmses, train_r2s = [], [], []
    val_losses, val_rmses, val_r2s = [], [], []
    test_losses, test_rmses, test_r2s = [], [], []
    for epoch in range(num_epochs):
        epochs.append(epoch)
        model.train()
        total_train_loss = 0
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss / len(train_loader))

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            y_pred_train = []
            y_true_train = []
            for data, targets in train_loader:
                outputs = model(data)  # 获取训练集预测结果
                y_pred_train.extend(outputs.numpy())  # 收集预测值
                y_true_train.extend(targets.numpy())  # 收集真实值
            train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))  # 计算训练集 RMSE
            train_r2 = r2_score(y_true_train, y_pred_train)  # 计算训练集 R2
            train_rmses.append(train_rmse)
            train_r2s.append(train_r2)

            # 对验证集进行评估
            y_pred_val = []
            y_true_val = []
            for data, targets in valid_loader:
                outputs = model(data)  # 获取测试集预测结果
                y_pred_val.extend(outputs.numpy())  # 收集预测值
                y_true_val.extend(targets.numpy())  # 收集真实值
            val_loss = np.sqrt(mean_squared_error(y_true_val, y_pred_val))  # 计算测试集 RMSE
            val_losses.append(val_loss)  # 记录测试损失
            val_rmse = np.sqrt(val_loss)  # 计算测试 RMSE
            val_r2 = r2_score(y_true_val, y_pred_val)  # 计算测试 R2
            val_rmses.append(val_rmse)
            val_r2s.append(val_r2)

            # 对测试集进行评估
            y_pred_test = []
            y_true_test = []
            for data, targets in test_loader:
                outputs = model(data)  # 获取测试集预测结果
                y_pred_test.extend(outputs.numpy())  # 收集预测值
                y_true_test.extend(targets.numpy())  # 收集真实值
            test_loss = np.sqrt(mean_squared_error(y_true_test, y_pred_test))  # 计算测试集 RMSE
            test_losses.append(test_loss)  # 记录测试损失
            test_rmse = np.sqrt(test_loss)  # 计算测试 RMSE
            test_r2 = r2_score(y_true_test, y_pred_test)  # 计算测试 R2
            test_rmses.append(test_rmse)
            test_r2s.append(test_r2)

        # 输出每个epoch的训练和测试结果
        print(
            f'Epoch {epoch + 1}, Train Loss: {total_train_loss / len(train_loader):.4f}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}, '
            f'val Loss: {val_loss:.4f}, val RMSE: {val_rmse:.4f}, val R2: {val_r2:.4f}, '
            f'Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}')

    # 将结果保存到DataFrame
    results_df = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Train RMSE': train_rmses,
        'Train R2': train_r2s,
        'Validation Loss': val_losses,
        'Validation RMSE': val_rmses,
        'Validation R2': val_r2s,
        'Test Loss': test_losses,
        'Test RMSE': test_rmses,
        'Test R2': test_r2s
    })

    # 保存到Excel文件
    results_df.to_excel('F:\game\小论文\paper1\结果\深度学习\\21材料\超参数优化\参数优化过程中的结果对比\\transformer\\training_validation_testing_results.xlsx', index=False)

    # 保存模型
    torch.save(model.state_dict(), 'F:\game\小论文\paper1\结果\深度学习\\21材料\超参数优化\参数优化过程中的结果对比\\transformer\\Best_TabTransformer_spectral_model.pth')
    print("模型已保存至文件 Best_TabTransformer_spectral_model.pth")

    return test_r2, test_rmse, test_loss


def evaluate_best_model(best_params, data_path, label_path):
    # 初始化模型
    model = OilContentPredictor(input_dim=256, model_dim=best_params['dim'], num_heads=4, num_layers=best_params['depth'],
                                output_dim=1, activation=best_params['activation'])

    num_epochs = 1800
    weight_decay = 0.001
    batch_size = best_params['batch_size']

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=weight_decay)

    train_loader, val_loader, test_loader = loadDataset(batch_size, data_path, label_path)

    test_r2, test_rmse, test_loss = best_model(model, optimizer, criterion, num_epochs, train_loader, val_loader, test_loader)

    # 返回评估指标
    return test_r2, test_rmse, test_loss


def main():
    data_path = r'F:\game\小论文\paper1\数据\21个材料\传统方法获取平均光谱.xlsx'
    label_path = r'F:\game\小论文\paper1\数据\21个材料\材料名与油分含量已聚类剔除按平均光谱排序21材料.xlsx'
    # 定义网格搜索空间参数
    param_grid = {
        'learning_rate': [0.00001, 0.000003, 0.000001],
        'batch_size': [8, 16, 32],
        'depth': [2, 3, 4],  # 网络层数
        'dim': [64, 128, 256],  # 隐藏单元数
        'activation': ["relu", "gelu"]  # 激活函数
    }

    # 生成所有参数组合
    all_params = product(*param_grid.values())

    best_r2 = float('inf')
    best_rmse = float('inf')
    best_loss = float('inf')
    best_params = None

    # 创建一个DataFrame用于存储每次训练结果
    results = pd.DataFrame(columns=['learning_rate', 'batch_size', 'depth', 'dim', 'activation', 'r2', 'rmse', 'loss'])

    for params in all_params:
        param_dict = dict(zip(param_grid.keys(), params))
        val_r2, val_rmse, val_loss = evaluate_model(param_dict, data_path, label_path)

        # 保存当前参数及对应的评估结果到DataFrame
        results = results.append({
            'learning_rate': param_dict['learning_rate'],
            'batch_size': param_dict['batch_size'],
            'depth': param_dict['depth'],
            'dim': param_dict['dim'],
            'activation': type(param_dict['activation']).__name__,
            'r2': val_r2,
            'rmse': val_rmse,
            'loss': val_loss
        }, ignore_index=True)

        # 更新最佳参数和评估指标
        if val_loss < best_loss:
            best_r2 = val_r2
            best_rmse = val_rmse
            best_loss = val_loss
            best_params = param_dict

    # 保存所有结果到Excel
    results.to_excel('F:\game\小论文\paper1\结果\深度学习\\21材料\超参数优化\参数优化过程中的结果对比\\transformer\\training_results.xlsx', index=False)

    # 评估最佳模型
    evaluate_best_model(best_params, data_path, label_path)

    print("最佳参数：", best_params)
    print("最佳得分：", best_r2, best_rmse, best_loss)


if __name__ == '__main__':
    main()