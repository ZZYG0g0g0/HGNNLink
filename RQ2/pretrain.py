import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import warnings

warnings.filterwarnings("ignore")

class FullDataset(Dataset):
    def __init__(self,
                 uc_folder,
                 cc_folder,
                 uc_bert_path,
                 cc_bert_path,
                 true_set_path):
        super().__init__()
        # 1) 获取文件夹中的所有文件，并排序
        self.uc_names = os.listdir(uc_folder)
        self.cc_names = os.listdir(cc_folder)
        # 2) 读取 BERT 向量 (DataFrame)
        uc_df = pd.read_excel(uc_bert_path, header=None, skiprows=1)
        cc_df = pd.read_excel(cc_bert_path, header=None, skiprows=1)
        self.uc_vectors = uc_df.values  # shape: (num_uc, 768)
        self.cc_vectors = cc_df.values  # shape: (num_cc, 768)

        # 3) 读取真集
        self.true_links = self._load_true_links(true_set_path)

        # 4) 构造所有 (需求名, 代码名) 对 & 标签（无下采样）
        self.all_pairs = []
        self.labels = []
        for uc_name in self.uc_names:
            for cc_name in self.cc_names:
                self.all_pairs.append((uc_name, cc_name))
                if (uc_name, cc_name) in self.true_links:
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        uc_name, cc_name = self.all_pairs[idx]
        label = self.labels[idx]

        uc_idx = self.uc_names.index(uc_name)
        cc_idx = self.cc_names.index(cc_name)

        uc_vec = self.uc_vectors[uc_idx]
        cc_vec = self.cc_vectors[cc_idx]

        concat_vec = torch.tensor(
            list(uc_vec) + list(cc_vec),
            dtype=torch.float
        )
        return concat_vec, torch.tensor(label, dtype=torch.float)

    def _load_true_links(self, true_set_path):
        true_links = set()
        with open(true_set_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uc_name, cc_name = line.split()
                true_links.add((uc_name, cc_name))
        return true_links


def undersample_train_dataset(dataset: FullDataset, neg_multiple=2):
    """
    对训练集中的正例、负例进行下采样，让正例:负例 = 1 : neg_multiple。
    :param dataset: 完整数据集(或其子集)，这里仅对其中的数据进行下采样
    :param neg_multiple: 若=2，则表示负例:正例=2:1 (即1:2的说法)。
    :return: 下采样后的 indices 列表
    """
    # 先拿到所有样本的下标
    indices = list(range(len(dataset)))
    # 分离正负例
    pos_indices = [i for i in indices if dataset.labels[i] == 1]
    neg_indices = [i for i in indices if dataset.labels[i] == 0]

    P = len(pos_indices)
    N = len(neg_indices)
    desired_neg_count = neg_multiple * P  # 如果 neg_multiple=2， 就是 2*P

    # 若负例数大于 desired_neg_count，则随机采样
    if P > 0 and N > desired_neg_count:
        neg_sampled = random.sample(neg_indices, desired_neg_count)
        final_indices = pos_indices + neg_sampled
        random.shuffle(final_indices)
        return final_indices
    else:
        # 否则保持原样(或者你可以选择别的策略)
        return indices


class SimpleModel(nn.Module):
    """
    一个最简单的网络结构 (不含BN等)，演示可行性
    """
    def __init__(self, input_dim=600):
        super().__init__()
        # 平均池化
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.bn_2 = nn.BatchNorm1d(input_dim // 2)
        # 全连接层
        self.fc1 = nn.Linear(input_dim // 2, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)  # 假设是二分类任务

    def forward(self, x):
        # 平均池化
        x = x.unsqueeze(1)  # 将 (batch_size, input_dim) 变为 (batch_size, 1, input_dim) 以便应用 1D 池化
        x = self.pool(x)
        x = x.squeeze(1)  # 恢复维度为 (batch_size, input_dim // 2)

        # 批量归一化和激活
        x = self.bn_2(x)
        x = torch.relu(x)

        # 全连接层及后续批量归一化和激活
        x = self.fc1(x)
        x = self.bn3(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = self.bn5(x)
        x = torch.relu(x)

        x = self.fc4(x)
        x = self.bn6(x)
        x = torch.relu(x)

        # 最后一层输出
        x = self.fc5(x)

        # 二分类概率输出
        x = torch.sigmoid(x)

        return x


def main(dataset):
    # dataset = "iTrust"
    uc_folder = f"../dataset/{dataset}/uc"
    cc_folder = f"../dataset/{dataset}/cc"
    uc_bert_path = f"../docs/{dataset}/uc/uc_word2vec_vectors.xlsx"
    cc_bert_path = f"../docs/{dataset}/cc/cc_word2vec_vectors.xlsx"
    true_set_path = f"../dataset/{dataset}/true_set.txt"

    # ========== 1) 构造完整数据集 ==========
    full_dataset = FullDataset(
        uc_folder=uc_folder,
        cc_folder=cc_folder,
        uc_bert_path=uc_bert_path,
        cc_bert_path=cc_bert_path,
        true_set_path=true_set_path
    )

    # 用于存储50次实验的结果
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    # 进行50次实验
    for trial in range(50):
        print(f"\n=== Trial {trial + 1} ===")

        # ========== 2) 9:1 随机拆分训练集、测试集 ==========
        full_indices = list(range(len(full_dataset)))
        random.shuffle(full_indices)  # 打乱所有下标

        split_point = int(0.9 * len(full_indices))  # 9:1
        train_indices = full_indices[:split_point]
        test_indices = full_indices[split_point:]

        # ========== 3) 对训练集做下采样 ==========
        undersampled_train_indices = undersample_train_dataset(
            dataset=full_dataset,
            neg_multiple=2
        )
        undersampled_train_indices = list(set(train_indices).intersection(undersampled_train_indices))

        # ========== 4) 构造 TrainDataset, TestDataset ==========
        train_dataset = Subset(full_dataset, undersampled_train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # ========== 5) 构造 DataLoader ==========
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

        # ========== 6) 定义模型、损失函数、优化器 ==========
        model = SimpleModel(input_dim=600)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # ========== 7) 训练 (示例: 30 epoch) ==========
        model.train()
        for epoch in range(30):
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch + 1}], Loss: {running_loss:.4f}", end='\r')

        # ========== 8) 在测试集上评估 ==========
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predicted = (outputs >= 0.5).long().squeeze(1)
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())

        # 计算当前实验的指标
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # 保存当前实验的结果
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)

    # ========== 9) 计算50次实验的平均指标 ==========
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores)

    print("\n=== Average Results Over 50 Trials ===")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")

    # ========== 10) 保存结果到 Excel ==========
    results_df = pd.DataFrame(
        {
            "Dataset": [dataset],
            "Precision": [avg_precision],
            "Recall": [avg_recall],
            "F1": [avg_f1],
        }
    )
    results_path = f"results_{dataset}.xlsx"
    results_df.to_excel(results_path, index=False)
    print(f"Results saved to '{results_path}'.")


if __name__ == "__main__":
    datasets = ["iTrust", "maven", "Derby", "Drools", "Seam2", "Pig", "Infinispan"]
    for dataset in datasets:
        print(dataset)
        main(dataset)

