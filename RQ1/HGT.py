import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 替换 HAN 类为 HGT 类
class HGTModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, metadata, heads=8, num_layers=2):
        super(HGTModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # HGT 通常包含多层 HGTConv
        for i in range(num_layers):
            conv = HGTConv(
                in_channels if i == 0 else hidden_channels,  # 输入维度为前一层的输出
                hidden_channels,
                metadata,
                heads=heads,  # 将 'num_heads' 改为 'heads'
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_channels))  # HGTConv 输出为 (hidden_channels * heads)

        #平均池化
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.bn_2 = nn.BatchNorm1d(hidden_channels // 2)
        # 全连接层
        self.fc1 = nn.Linear(hidden_channels // 2, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)  # 假设是二分类任务

    def forward(self, x_dict, edge_index_dict):
        for conv, bn in zip(self.convs, self.bns):
            x_dict = conv(x_dict, edge_index_dict)
            for key in x_dict.keys():
                x = x_dict[key]
                x = bn(x)
                x = F.relu(x)
                x_dict[key] = x

        for key in x_dict.keys():
            x = x_dict[key]

            # 池化操作
            x = x.unsqueeze(1)  # 将 (batch_size, num_features) 变为 (batch_size, 1, num_features) 以便应用 1D 池化
            x = self.pool(x)
            x = x.squeeze(1)  # 变回 (batch_size, num_features/2)

            # 批量归一化和激活
            x = self.bn_2(x)
            x = F.relu(x)

            # 全连接层及后续批量归一化和激活
            x = self.fc1(x)
            x = self.bn3(x)
            x = F.relu(x)

            x = self.fc2(x)
            x = self.bn4(x)
            x = F.relu(x)

            x = self.fc3(x)
            x = self.bn5(x)
            x = F.relu(x)

            x = self.fc4(x)
            x = self.bn6(x)
            x = F.relu(x)

            x = self.fc5(x)

            x_dict[key] = x

        return x_dict

class Classifier(torch.nn.Module):
    def forward(self, x_req: Tensor, x_code: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_req = x_req[edge_label_index[0]]
        edge_feat_code = x_code[edge_label_index[1]]
        return torch.sigmoid((edge_feat_req * edge_feat_code).sum(dim=-1))

class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata):
        super(Model, self).__init__()
        self.hgt = HGTModel(in_channels, out_channels, metadata)  # 使用 HGTModel 替代 HAN
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "req": data["req"].x,
            "code": data["code"].x,
        }
        x_dict = self.hgt(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["req"],
            x_dict["code"],
            data["req", "link", "code"].edge_label_index,
        )
        return pred

def generate_req_code_edge(dataset_name):
    uc_names = os.listdir('../dataset/' + dataset_name + '/uc')
    cc_names = os.listdir('../dataset/' + dataset_name + '/cc')
    uc_idx_dict = {uc_names[i]: i for i in range(len(uc_names))}
    cc_idx_dict = {cc_names[j].split('.')[0]: j for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    with open('../dataset/' + dataset_name + "/true_set.txt", 'r', encoding='ISO8859-1') as df:
        lines = df.readlines()
    for line in lines:
        uc_name, cc_name = line.split(' ')[0], line.split(' ')[1].split('.')[0]
        if uc_name in uc_idx_dict and cc_name in cc_idx_dict:
            edge_from.append(uc_idx_dict[uc_name])
            edge_to.append(cc_idx_dict[cc_name])
    return edge_from, edge_to

# 生成继承边
def generate_extend_edges(dataset_name):
    cc_names = os.listdir('../dataset/' + dataset_name + '/cc')
    cc_idx_dict = {cc_names[j].split('.')[0]: j for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    extend_df = pd.read_excel(f'../docs/{dataset_name}/cc/{dataset_name}ClassRelationships.xlsx')
    for _, row in extend_df.iterrows():
        cc_name1, cc_name2, relationship = row['Class 1'], row['Class 2'], row['Relationship']
        if cc_name1 in cc_idx_dict and cc_name2 in cc_idx_dict:
            if relationship == 'extend':
                edge_from.append(cc_idx_dict[cc_name1])
                edge_to.append(cc_idx_dict[cc_name2])
    return edge_from, edge_to

# 生成调用边
def generate_import_edges(dataset_name):
    cc_names = os.listdir('../dataset/' + dataset_name + '/cc')
    cc_idx_dict = {cc_names[j].split('.')[0]: j for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    extend_df = pd.read_excel(f'../docs/{dataset_name}/cc/{dataset_name}ClassRelationships.xlsx')
    for _, row in extend_df.iterrows():
        cc_name1, cc_name2, relationship = row['Class 1'], row['Class 2'], row['Relationship']
        if cc_name1 in cc_idx_dict and cc_name2 in cc_idx_dict:
            if relationship == 'import':
                edge_from.append(cc_idx_dict[cc_name1])
                edge_to.append(cc_idx_dict[cc_name2])
    return edge_from, edge_to

def generate_IR_edges(dataset_name):
    # 获取需求和代码的文件名列表
    uc_names = os.listdir(f'../dataset/{dataset_name}/uc')
    cc_names = os.listdir(f'../dataset/{dataset_name}/cc')

    # 创建索引字典
    uc_idx_dict = {uc_names[i]: i for i in range(len(uc_names))}
    cc_idx_dict = {cc_names[j].split('.')[0]: j for j in range(len(cc_names))}

    # 读取相似度数据
    data = pd.read_excel(f'../docs/{dataset_name}/IR_feature.xlsx')

    # 对每一列进行排名
    ranked_data = data[['vsm_1', 'vsm_2', 'lsi_1', 'lsi_2', 'lda_1', 'lda_2', 'bm25_1', 'bm25_2', 'JS_1', 'JS_2']].rank(
        method='average')

    # 计算平均排名
    ranked_data['avg_rank'] = ranked_data.mean(axis=1)
    # 组合 code, requirement 和 avg_rank 列
    combined_data = pd.DataFrame({
        'requirement': data['requirement'],
        'code': data['code'],
        'avg_rank': ranked_data['avg_rank']
    })
    # 选择前50%的链接
    top_10_percent_threshold = combined_data['avg_rank'].quantile(0.5)
    top_links = combined_data[combined_data['avg_rank'] <= top_10_percent_threshold]

    # 生成边关系
    edge_from = [uc_idx_dict[requirement] for requirement in top_links['requirement']]
    edge_to = [cc_idx_dict[code] for code in top_links['code']]

    return edge_from, edge_to

if __name__ == '__main__':
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 定义要遍历的数据集和节点特征

    datasets = ['Infinispan', 'iTrust', 'Maven', 'Pig', 'Seam2','Derby', 'Drools']
    nodes_features = ['word2vec', 'bert', 'albert', 'roberta', 'xlnet']

    # 定义要遍历的边类型组合
    edge_type_combinations = [
        ('None',),
        ('IR',),
        ('extend',),
        ('import',),
        ('IR', 'extend'),
        ('IR', 'import'),
        ('import', 'extend'),
        ('IR', 'extend', 'import')
    ]

    # 准备一个总的结果列表
    all_results = []

    # 创建结果保存目录
    output_dir = './result/'
    os.makedirs(output_dir, exist_ok=True)

    for nodes_feature in nodes_features:
        for dataset in datasets:
            # 读取节点特征
            uc_df = pd.read_excel(f'../docs/{dataset}/uc/uc_{nodes_feature}_vectors.xlsx')
            cc_df = pd.read_excel(f'../docs/{dataset}/cc/cc_{nodes_feature}_vectors.xlsx')
            req_feat = torch.from_numpy(uc_df.values).to(torch.float)
            code_feat = torch.from_numpy(cc_df.values).to(torch.float)

            # 生成基础边（req-link-code）
            edge_from, edge_to = generate_req_code_edge(dataset)
            edge_index = torch.tensor([edge_from, edge_to], dtype=torch.long)

            # 生成所有可能的边
            extend_edge_from, extend_edge_to = generate_extend_edges(dataset)
            import_edge_from, import_edge_to = generate_import_edges(dataset)
            IR_edge_from, IR_edge_to = generate_IR_edges(dataset)
            extend_edge_index = torch.tensor([extend_edge_from, extend_edge_to], dtype=torch.long)
            import_edge_index = torch.tensor([import_edge_from, import_edge_to], dtype=torch.long)
            IR_edge_index = torch.tensor([IR_edge_from, IR_edge_to], dtype=torch.long)

            # 遍历边类型组合
            for edge_types in edge_type_combinations:
                # 初始化HeteroData
                data = HeteroData()
                data["req"].x = req_feat
                data["code"].x = code_feat
                data["req", "link", "code"].edge_index = edge_index

                # 根据当前组合添加边
                if 'extend' in edge_types:
                    data["code", "extend", "code"].edge_index = extend_edge_index
                if 'import' in edge_types:
                    data["code", "import", "code"].edge_index = import_edge_index
                if 'IR' in edge_types:
                    data["req", "sim", "code"].edge_index = IR_edge_index

                # 将边转换为无向
                data = T.ToUndirected()(data)

                # 初始化分数列表
                precision_scores = []
                recall_scores = []
                f1_scores = []

                # 重复实验50次
                for i in range(50):
                    print(f"Dataset: {dataset}, Node Feature: {nodes_feature}, Edge Types: {edge_types}, Experiment {i+1}/50")
                    transform = T.RandomLinkSplit(
                        num_test=0.1,  # 测试集比例为10%
                        disjoint_train_ratio=0.3, 
                        neg_sampling_ratio=2.0, 
                        add_negative_train_samples=False, 
                        edge_types=("req", "link", "code"), 
                        rev_edge_types=("code", "rev_link", "req")
                    )
                    train_data, _, test_data = transform(data)  # 忽略验证集

                    train_loader = LinkNeighborLoader(
                        data=train_data,
                        num_neighbors=[20, 10],
                        neg_sampling_ratio=2.0,
                        edge_label_index=(("req", "link", "code"), train_data["req", "link", "code"].edge_label_index),
                        edge_label=train_data["req", "link", "code"].edge_label,
                        batch_size=128,
                        shuffle=True,
                    )
                    model = Model(in_channels=req_feat.size(1),  out_channels=128, metadata=data.metadata())
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                    for epoch in range(1, 30):
                        total_loss = total_examples = 0
                        for sampled_data in train_loader:
                            try:
                                optimizer.zero_grad()
                                sampled_data = sampled_data.to(device)
                                pred = model(sampled_data)
                                ground_truth = sampled_data["req", "link", "code"].edge_label
                                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                                loss.backward()
                                optimizer.step()
                                total_loss += float(loss) * pred.numel()
                                total_examples += pred.numel()
                            except IndexError as e:
                                print(f"IndexError: {e}")
                                print(f"Sampled Data: {sampled_data}")
                                break

                    test_loader = LinkNeighborLoader(
                        data=test_data,
                        num_neighbors=[20, 10],
                        edge_label_index=(("req", "link", "code"), test_data["req", "link", "code"].edge_label_index),
                        edge_label=test_data["req", "link", "code"].edge_label,
                        batch_size=3 * 128,
                        shuffle=False,
                    )

                    preds = []
                    ground_truths = []
                    for sampled_data in test_loader:
                        with torch.no_grad():
                            sampled_data = sampled_data.to(device)
                            preds.append(model(sampled_data).cpu())
                            ground_truths.append(sampled_data["req", "link", "code"].edge_label.cpu())
                    pred = torch.cat(preds, dim=0).numpy()
                    ground_truth = torch.cat(ground_truths, dim=0).numpy()
                    pred_labels = (pred > 0.5).astype(np.float32)

                    precision = precision_score(ground_truth, pred_labels, average='binary')
                    recall = recall_score(ground_truth, pred_labels, average='binary')
                    f1 = f1_score(ground_truth, pred_labels, average='binary')

                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)

                # 计算平均值
                avg_precision = np.mean(precision_scores)
                avg_recall = np.mean(recall_scores)
                avg_f1 = np.mean(f1_scores)

                print(f"Average Precision: {avg_precision:.4f}")
                print(f"Average Recall: {avg_recall:.4f}")
                print(f"Average F1: {avg_f1:.4f}")

                # 记录结果
                all_results.append({
                    'Dataset': dataset,
                    'Node Feature': nodes_feature,
                    'Edge Types': '+'.join(edge_types),
                    'Precision': avg_precision,
                    'Recall': avg_recall,
                    'F1': avg_f1
                })

    # 将所有结果写入Excel
    results_df = pd.DataFrame(all_results)
    results_df.to_excel('./result/hgt_rq1.xlsx', index=False)
    print("All experiments completed. Results saved to './result/hgt_rq1.xlsx'.")
