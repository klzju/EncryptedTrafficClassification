import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 模拟数据
def generate_data(num_samples=1000, seq_length=1500, num_classes=3):
    X = np.random.rand(num_samples, seq_length).astype(np.float32)  # 模拟流量特征
    y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)  # 模拟标签
    return X, y

# 自定义数据集
class TrafficDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data).unsqueeze(1)  # 添加通道维度 (N, 1, seq_length)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 生成数据
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据加载
train_dataset = TrafficDataset(X_train, y_train)
test_dataset = TrafficDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
