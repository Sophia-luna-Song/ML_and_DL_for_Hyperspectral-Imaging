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
from itertools import product


def loadDataset(batch_size, data_path, label_path):
    # 加载数据
    data_df = pd.read_excel(data_path, sheet_name='Sheet1', header=None)
    data1 = data_df.iloc[1:346, 0:257]
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
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    # 处理离散特征
    label_encoder = LabelEncoder()
    all_categ_features = np.concatenate([data[:, 0], data[:, 0]])
    label_encoder.fit(all_categ_features)

    x_categ_train = label_encoder.transform(X_train[:, 0])
    x_categ_test = label_encoder.transform(X_test[:, 0])
    x_categ_valid = label_encoder.transform(X_val[:, 0])
    x_categ_valid = torch.tensor(x_categ_valid).long().unsqueeze(1)
    x_categ_train = torch.tensor(x_categ_train).long().unsqueeze(1)
    x_categ_test = torch.tensor(x_categ_test).long().unsqueeze(1)

    # 处理数值特征
    x_cont_train = torch.tensor(X_train[:, 1:]).float()
    x_cont_valid = torch.tensor(X_val[:, 1:], dtype=torch.float32)
    x_cont_test = torch.tensor(X_test[:, 1:]).float()

    # Convert labels to tensors
    y_train_tensor = torch.tensor(y_train).float().unsqueeze(1)
    y_valid_tensor = torch.tensor(y_val).float().unsqueeze(1)
    y_test_tensor = torch.tensor(y_test).float().unsqueeze(1)

    # Data loader
    train_dataset = TensorDataset(x_categ_train, x_cont_train, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = TensorDataset(x_categ_valid, x_cont_valid, y_valid_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(x_categ_test, x_cont_test, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, validation_loader


def modelTab():
    model = TabTransformer(
        categories=[22],  # tuple containing the number of unique values within each category
        num_continuous=256,  # number of continuous values
        dim=32,  # dimension, paper set at 32
        dim_out=1,  # binary prediction, but could be anything
        depth=6,  # depth, paper recommended 6
        heads=8,  # heads, paper recommends 8
        attn_dropout=0.1,  # post-attention dropout
        ff_dropout=0.1,  # feed forward dropout
        mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=nn.ReLU()  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    )

    # Assume you have the following dummy input for categorical and continuous data
    x_categ_dummy = torch.randint(0, 22, (1, 1)).long()  # One categorical feature
    x_cont_dummy = torch.randn(1, 256)  # 256 continuous features

    # Visualize the model
    y_dummy = model(x_categ_dummy, x_cont_dummy)
    dot = torchviz.make_dot(y_dummy, params=dict(
        list(model.named_parameters()) + [('x_categ', x_categ_dummy), ('x_cont', x_cont_dummy)]))
    dot.render('TabTransformer_model_visualization', format='png')


    return model


def train(model, optimizer, criterion, num_epochs, train_loader, test_loader):
    # Training and evaluation
    train_losses = []
    test_losses = []
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
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            y_pred_train = []
            y_true_train = []
            for x_categ, x_cont, y in train_loader:
                outputs = model(x_categ, x_cont)  # 获取训练集预测结果
                y_pred_train.extend(outputs.numpy())  # 收集预测值
                y_true_train.extend(y.numpy())  # 收集真实值
            train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))  # 计算训练集 RMSE
            train_r2 = r2_score(y_true_train, y_pred_train)  # 计算训练集 R2

            # 对测试集进行评估
            y_pred_test = []
            y_true_test = []
            for x_categ, x_cont, y in test_loader:
                outputs = model(x_categ, x_cont)  # 获取测试集预测结果
                y_pred_test.extend(outputs.numpy())  # 收集预测值
                y_true_test.extend(y.numpy())  # 收集真实值
            test_loss = np.sqrt(mean_squared_error(y_true_test, y_pred_test))  # 计算测试集 RMSE
            test_losses.append(test_loss)  # 记录测试损失
            test_rmse = np.sqrt(test_loss)  # 计算测试 RMSE
            test_r2 = r2_score(y_true_test, y_pred_test)  # 计算测试 R2

        # 输出每个epoch的训练和测试结果
        print(
            f'Epoch {epoch + 1}, Train Loss: {total_train_loss / len(train_loader):.4f}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}, Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'TabTransformer_spectral_model.pth')
    print("模型已保存至文件 TabTransformer_spectral_model.pth")
    # Visualization of training and testing losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss per Epoch')
    plt.legend()
    plt.show()
    return model


# 网格搜索 超参数验证主体
def validT(model, optimizer, criterion, epochs, train_loader, val_loader):
    # Training and evaluation
    train_losses, train_rmses, train_r2s = [], [], []
    val_losses, val_rmses, val_r2s = [], [], []
    for epoch in range(epochs):
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

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            y_pred_train = []
            y_true_train = []
            for x_categ, x_cont, y in train_loader:
                outputs = model(x_categ, x_cont)  # 获取训练集预测结果
                y_pred_train.extend(outputs.numpy())  # 收集预测值
                y_true_train.extend(y.numpy())  # 收集真实值
            train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))  # 计算训练集 RMSE
            train_r2 = r2_score(y_true_train, y_pred_train)  # 计算训练集 R2
            train_rmses.append(train_rmse)
            train_r2s.append(train_r2)

            # 对测试集进行评估
            y_pred_test = []
            y_true_test = []
            for x_categ, x_cont, y in val_loader:
                outputs = model(x_categ, x_cont)  # 获取测试集预测结果
                y_pred_test.extend(outputs.numpy())  # 收集预测值
                y_true_test.extend(y.numpy())  # 收集真实值
            val_loss = np.sqrt(mean_squared_error(y_true_test, y_pred_test))  # 计算测试集 RMSE
            val_losses.append(val_loss)  # 记录测试损失
            val_rmse = np.sqrt(val_loss)  # 计算测试 RMSE
            val_r2 = r2_score(y_true_test, y_pred_test)  # 计算测试 R2
            val_rmses.append(val_rmse)
            val_r2s.append(val_r2)

        # 输出每个epoch的训练和测试结果
        print(
            f'Epoch {epoch + 1}, Train Loss: {total_train_loss / len(train_loader):.4f}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}')
    # # 将结果保存到DataFrame
    # results_df = pd.DataFrame({
    #     'Epoch': epochs,
    #     'Train Loss': train_losses,
    #     'Train RMSE': train_rmses,
    #     'Train R2': train_r2s,
    #     'Test Loss': val_losses,
    #     'Test RMSE': val_rmses,
    #     'Test R2': val_r2s,
    #     'Validation Loss': val_losses,
    #     'Validation RMSE': val_rmses,
    #     'Validation R2': val_r2s
    # })
    #
    # # 保存到Excel文件
    # results_df.to_excel(f'F:\game\小论文\paper1\结果\深度学习\\超参数优化_results.xlsx', index=False)

    return val_r2, val_rmse, val_loss



def evaluate_model(params, data_path, label_path):
    # 初始化模型
    model = TabTransformer(
        categories=[21],  # 固定值
        num_continuous=256,  # 固定值
        dim=params['dim'],
        dim_out=1,
        depth=params['depth'],
        heads=8,  # 可以调整
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=params['activation']
    )

    num_epochs = 100
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
        for x_categ, x_cont, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x_categ, x_cont)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss / len(train_loader))

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            y_pred_train = []
            y_true_train = []
            for x_categ, x_cont, y in train_loader:
                outputs = model(x_categ, x_cont)  # 获取训练集预测结果
                y_pred_train.extend(outputs.numpy())  # 收集预测值
                y_true_train.extend(y.numpy())  # 收集真实值
            train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))  # 计算训练集 RMSE
            train_r2 = r2_score(y_true_train, y_pred_train)  # 计算训练集 R2
            train_rmses.append(train_rmse)
            train_r2s.append(train_r2)

            # 对验证集进行评估
            y_pred_val = []
            y_true_val = []
            for x_categ, x_cont, y in valid_loader:
                outputs = model(x_categ, x_cont)  # 获取测试集预测结果
                y_pred_val.extend(outputs.numpy())  # 收集预测值
                y_true_val.extend(y.numpy())  # 收集真实值
            val_loss = np.sqrt(mean_squared_error(y_true_val, y_pred_val))  # 计算测试集 RMSE
            val_losses.append(val_loss)  # 记录测试损失
            val_rmse = np.sqrt(val_loss)  # 计算测试 RMSE
            val_r2 = r2_score(y_true_val, y_pred_val)  # 计算测试 R2
            val_rmses.append(val_rmse)
            val_r2s.append(val_r2)

            # 对测试集进行评估
            y_pred_test = []
            y_true_test = []
            for x_categ, x_cont, y in test_loader:
                outputs = model(x_categ, x_cont)  # 获取测试集预测结果
                y_pred_test.extend(outputs.numpy())  # 收集预测值
                y_true_test.extend(y.numpy())  # 收集真实值
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
    results_df.to_excel('F:\game\小论文\paper1\结果\深度学习\\21材料\超参数优化\参数优化过程中的结果对比\\training_validation_testing_results.xlsx', index=False)

    # 保存模型
    torch.save(model.state_dict(), 'F:\game\小论文\paper1\结果\深度学习\\21材料\超参数优化\参数优化过程中的结果对比\\128dim——Best_TabTransformer_spectral_model.pth')
    print("模型已保存至文件 Best_TabTransformer_spectral_model.pth")

    return test_r2, test_rmse, test_loss


def evaluate_best_model(best_params, data_path, label_path):
    # 初始化模型
    model = TabTransformer(
        categories=[21],  # 固定值
        num_continuous=256,  # 固定值
        dim=best_params['dim'],
        dim_out=1,
        depth=best_params['depth'],
        heads=8,  # 可以调整
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=best_params['activation']
    )

    num_epochs = 100
    weight_decay = 0.001
    batch_size = best_params['batch_size']

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=weight_decay)

    train_loader, val_loader, test_loader = loadDataset(batch_size, data_path, label_path)

    test_r2, test_rmse, test_loss = best_model(model, optimizer, criterion, num_epochs, train_loader, val_loader, test_loader)

    # 返回评估指标
    return test_r2, test_rmse, test_loss


def main():
    modelTab()
    data_path = r'F:\game\小论文\paper1\数据\21个材料\传统方法获取平均光谱.xlsx'
    label_path = r'F:\game\小论文\paper1\数据\21个材料\材料名与油分含量已聚类剔除按平均光谱排序21材料.xlsx'
    # 定义网格搜索空间参数
    param_grid = {
        'learning_rate': [0.00001, 0.000003, 0.000001],
        'batch_size': [8, 16, 32],
        'depth': [2, 4, 6],  # 网络层数
        'dim': [16, 32, 64],  # 隐藏单元数
        'activation': [nn.ReLU(), nn.LeakyReLU(), nn.SELU()]  # 激活函数
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
    results.to_excel('F:\game\小论文\paper1\结果\深度学习\\21材料\超参数优化\参数优化过程中的结果对比\\training_results.xlsx', index=False)

    # 评估最佳模型
    evaluate_best_model(best_params, data_path, label_path)

    print("最佳参数：", best_params)
    print("最佳得分：", best_r2, best_rmse, best_loss)


if __name__ == '__main__':
    main()