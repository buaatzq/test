import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
import numpy as np

def SoftMax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim = 1,keepdim = True) #dim=1时为按行求和，dim=0时为按列求和
    return X_exp/partition

# 获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义和初始化模型
num_inputs = 28*28
num_outputs = 10

W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)
b = torch.zeros(num_outputs,dtype=torch.float)

# 保留对应的梯度信息
W.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)

X = torch.rand(2,5)
X = SoftMax(X)
print(X)










