import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 提取特征函数
def extract_features(data):
    texts = [entry['texts'][0] for entry in data]  # 取出texts字段中的文本
    original_texts = [entry['original_text'] for entry in data]
    titles = [entry['title'] for entry in data]

    # 特征提取示例：使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    texts_features = vectorizer.fit_transform(texts).toarray()
    original_texts_features = vectorizer.transform(original_texts).toarray()
    titles_features = vectorizer.transform(titles).toarray()

    # 合并所有特征
    features = np.concatenate((texts_features, original_texts_features, titles_features), axis=1)

    return features

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # 将标签转换为列向量

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 加载数据集
train_data = load_data('MLP_train.json')
test_data = load_data('MLP_test.json')
valid_data = load_data('MLP_valid.json')

# 提取特征
X_train = extract_features(train_data)
y_train = [entry['label'] for entry in train_data]
X_test = extract_features(test_data)
y_test = [entry['label'] for entry in test_data]
X_valid = extract_features(valid_data)
y_valid = [entry['label'] for entry in valid_data]

# 使用PCA将特征降到2维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
X_valid_pca = pca.transform(X_valid)

# 创建数据集和数据加载器
train_dataset = CustomDataset(X_train_pca, y_train)
test_dataset = CustomDataset(X_test_pca, y_test)
valid_dataset = CustomDataset(X_valid_pca, y_valid)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# 初始化模型、损失函数和优化器
input_size = 2
hidden_size = 64
output_size = 1
model = MLP(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    best_valid_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # 在验证集上评估模型
        model.eval()
        valid_preds = []
        valid_labels = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())

        valid_accuracy = accuracy_score(valid_labels, valid_preds)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader.dataset)}, Valid Accuracy: {valid_accuracy}')

        # 保存最佳模型
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), 'best_mlp_model.pth')
            print("Saved the best model with validation accuracy:", best_valid_accuracy)

# 训练模型
train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10)

# 在测试集上评估模型
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
print(f'Test Accuracy: {test_accuracy}')
