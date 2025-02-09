# -*- coding:utf-8 -*-

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class RGCN_LP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations):
        super(RGCN_LP, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations, num_bases=30)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations, num_bases=30)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # 池化层
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.bn_pool = nn.BatchNorm1d(hidden_channels)

        # 全连接层
        self.fc1 = nn.Linear(hidden_channels, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)

    def encode(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = self.bn2(x)
        x = F.relu(x)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        combined = torch.cat([src, dst], dim=-1)
        combined = combined.unsqueeze(1)  # [batch_size, 1, 2 * hidden_channels]
        combined = self.pool(combined)     # [batch_size, 1, hidden_channels]
        combined = combined.squeeze(1)     # [batch_size, hidden_channels]

        expected_features = self.bn_pool.num_features
        actual_features = combined.size(1)
        assert actual_features == expected_features, f"Expected combined to have {expected_features} features, but got {actual_features}"

        combined = self.bn_pool(combined)
        combined = F.relu(combined)
        combined = self.fc1(combined)
        combined = self.bn3(combined)
        combined = F.relu(combined)

        combined = self.fc2(combined)
        combined = self.bn4(combined)
        combined = F.relu(combined)

        combined = self.fc3(combined)
        combined = self.bn5(combined)
        combined = F.relu(combined)

        combined = self.fc4(combined)
        combined = self.bn6(combined)
        combined = F.relu(combined)

        out = self.fc5(combined)
        return out

    def forward(self, x, edge_index, edge_type, edge_label_index):
        z = self.encode(x, edge_index, edge_type)
        return self.decode(z, edge_label_index)

# 辅助函数部分保持不变
def generate_req_code_edge(dataset_name):
    uc_names = os.listdir('../dataset/' + dataset_name + '/uc')
    cc_names = os.listdir('../dataset/' + dataset_name + '/cc')
    uc_idx_dict = {uc_names[i]: i for i in range(len(uc_names))}
    cc_idx_dict = {cc_names[j].split('.')[0]: j + len(uc_names) for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    with open('../dataset/' + dataset_name + "/true_set.txt", 'r', encoding='ISO8859-1') as df:
        lines = df.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) < 2:
            continue  # 跳过格式不正确的行
        uc_name, cc_name = parts[0], parts[1].split('.')[0]
        if uc_name in uc_idx_dict and cc_name in cc_idx_dict:
            edge_from.append(uc_idx_dict[uc_name])
            edge_to.append(cc_idx_dict[cc_name])
    return edge_from, edge_to

def generate_extend_edges(dataset_name, num_req):
    cc_names = os.listdir('../dataset/' + dataset_name + '/cc')
    cc_idx_dict = {cc_names[j].split('.')[0]: j + num_req for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    extend_df = pd.read_excel(f'../docs/{dataset_name}/cc/{dataset_name}ClassRelationships.xlsx')
    for _, row in extend_df.iterrows():
        cc_name1, cc_name2, relationship = row['Class 1'], row['Class 2'], row['Relationship']
        if cc_name1 in cc_idx_dict and cc_name2 in cc_idx_dict:
            if relationship == 'extend':
                edge_from.append(cc_idx_dict[cc_name1])
                edge_to.append(cc_idx_dict[cc_name2])
    return edge_from, edge_to

def generate_import_edges(dataset_name, num_req):
    cc_names = os.listdir('../dataset/' + dataset_name + '/cc')
    cc_idx_dict = {cc_names[j].split('.')[0]: j + num_req for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    extend_df = pd.read_excel(f'../docs/{dataset_name}/cc/{dataset_name}ClassRelationships.xlsx')
    for _, row in extend_df.iterrows():
        cc_name1, cc_name2, relationship = row['Class 1'], row['Class 2'], row['Relationship']
        if cc_name1 in cc_idx_dict and cc_name2 in cc_idx_dict:
            if relationship == 'import':
                edge_from.append(cc_idx_dict[cc_name1])
                edge_to.append(cc_idx_dict[cc_name2])
    return edge_from, edge_to

def generate_IR_edges(dataset_name, num_req):
    uc_names = os.listdir(f'../dataset/{dataset_name}/uc')
    cc_names = os.listdir(f'../dataset/{dataset_name}/cc')

    uc_idx_dict = {uc_names[i]: i for i in range(len(uc_names))}
    cc_idx_dict = {cc_names[j].split('.')[0]: j + num_req for j in range(len(cc_names))}

    data = pd.read_excel(f'../docs/{dataset_name}/IR_feature.xlsx')

    ranked_data = data[['vsm_1', 'vsm_2', 'lsi_1', 'lsi_2', 'lda_1', 'lda_2', 'bm25_1', 'bm25_2', 'JS_1', 'JS_2']].rank(
        method='average')

    ranked_data['avg_rank'] = ranked_data.mean(axis=1)
    combined_data = pd.DataFrame({
        'requirement': data['requirement'],
        'code': data['code'],
        'avg_rank': ranked_data['avg_rank']
    })
    top_50_percent_threshold = combined_data['avg_rank'].quantile(0.5)
    top_links = combined_data[combined_data['avg_rank'] <= top_50_percent_threshold]

    edge_from = [uc_idx_dict[requirement] for requirement in top_links['requirement'] if requirement in uc_idx_dict]
    edge_to = [cc_idx_dict[code] for code in top_links['code'] if code in cc_idx_dict]

    return edge_from, edge_to

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    datasets = ['Infinispan','iTrust', 'maven', 'Pig', 'Seam2' , 'Derby', 'Drools']
    nodes_features = ['xlnet', 'word2vec', 'bert', 'albert', 'roberta']
    # 定义边类型组合，确保每个组合都包含 'link'
    edge_type_combinations = [
        ['link'],
        ['link', 'sim'],
        ['link', 'extend'],
        ['link', 'import'],
        ['link', 'extend', 'import'],
        ['link', 'sim', 'import'],
        ['link', 'sim', 'extend'],
        ['link', 'sim', 'extend', 'import'],
    ]

    # 边类型映射，包括反向边类型
    edge_type_mapping = {
        'link': 0,
        'extend': 1,
        'import': 2,
        'sim': 3,
        'rev_link': 4,
        'rev_extend': 5,
        'rev_import': 6,
        'rev_sim': 7
    }
    num_relations = len(edge_type_mapping)  # 8

    all_results = []

    for nodes_feature in nodes_features:
        for dataset in datasets:
            # 读取节点特征
            uc_df = pd.read_excel(f'../docs/{dataset}/uc/uc_{nodes_feature}_vectors.xlsx')
            cc_df = pd.read_excel(f'../docs/{dataset}/cc/cc_{nodes_feature}_vectors.xlsx')
            req_feat = torch.from_numpy(uc_df.values).to(torch.float)
            code_feat = torch.from_numpy(cc_df.values).to(torch.float)
            x = torch.cat([req_feat, code_feat], dim=0)
            num_req = req_feat.size(0)

            # 生成各类型边
            link_from, link_to = generate_req_code_edge(dataset)
            extend_from, extend_to = generate_extend_edges(dataset, num_req)
            import_from, import_to = generate_import_edges(dataset, num_req)
            IR_from, IR_to = generate_IR_edges(dataset, num_req)

            # 转换为张量
            link_from = torch.tensor(link_from, dtype=torch.long)
            link_to = torch.tensor(link_to, dtype=torch.long)
            extend_from = torch.tensor(extend_from, dtype=torch.long)
            extend_to = torch.tensor(extend_to, dtype=torch.long)
            import_from = torch.tensor(import_from, dtype=torch.long)
            import_to = torch.tensor(import_to, dtype=torch.long)
            IR_from = torch.tensor(IR_from, dtype=torch.long)
            IR_to = torch.tensor(IR_to, dtype=torch.long)

            # 创建所有边的Data对象
            all_edge_index = torch.cat([
                torch.stack([link_from, link_to], dim=0),
                torch.stack([extend_from, extend_to], dim=0),
                torch.stack([import_from, import_to], dim=0),
                torch.stack([IR_from, IR_to], dim=0)
            ], dim=1)

            all_edge_type = torch.cat([
                torch.full((link_from.size(0),), edge_type_mapping['link'], dtype=torch.long),
                torch.full((extend_from.size(0),), edge_type_mapping['extend'], dtype=torch.long),
                torch.full((import_from.size(0),), edge_type_mapping['import'], dtype=torch.long),
                torch.full((IR_from.size(0),), edge_type_mapping['sim'], dtype=torch.long)
            ], dim=0)

            num_nodes = x.size(0)
            max_edge = all_edge_index.max().item()
            assert max_edge < num_nodes, "边的索引超出节点数量"

            data = Data(x=x, edge_index=all_edge_index, edge_type=all_edge_type)
            data = T.ToUndirected()(data)

            # 单独提取“link”边
            link_mask = (data.edge_type == edge_type_mapping['link'])
            link_edge_index = data.edge_index[:, link_mask]
            link_edge_type = data.edge_type[link_mask]


            # 创建仅包含“link”边的Data对象
            link_data = Data(x=data.x, edge_index=link_edge_index, edge_type=link_edge_type)
            link_data = T.ToUndirected()(link_data)

            # 拆分“link”边为训练集和测试集
            transform = T.RandomLinkSplit(
                num_test=0.1,  # 测试集占10%
                is_undirected=True,
                add_negative_train_samples=False
            )
            train_link, _, test_link = transform(link_data)

            # 创建测试集Data对象（仅包含测试“link”边）
            test_data = test_link

            # 遍历不同的边类型组合
            for edge_types in edge_type_combinations:
                print(f"Processing combination: {edge_types}")

                # 初始化训练集边
                train_edge_index = train_link.edge_index.clone()
                train_edge_type = train_link.edge_type.clone()

                # 根据当前组合添加其他边类型
                if 'sim' in edge_types:
                    sim_mask = (data.edge_type == edge_type_mapping['sim'])
                    sim_edge_index = data.edge_index[:, sim_mask]
                    sim_edge_type = data.edge_type[sim_mask]
                    train_edge_index = torch.cat([train_edge_index, sim_edge_index], dim=1)
                    train_edge_type = torch.cat([train_edge_type, sim_edge_type], dim=0)

                if 'extend' in edge_types:
                    extend_mask = (data.edge_type == edge_type_mapping['extend'])
                    extend_edge_index = data.edge_index[:, extend_mask]
                    extend_edge_type = data.edge_type[extend_mask]
                    train_edge_index = torch.cat([train_edge_index, extend_edge_index], dim=1)
                    train_edge_type = torch.cat([train_edge_type, extend_edge_type], dim=0)

                if 'import' in edge_types:
                    import_mask = (data.edge_type == edge_type_mapping['import'])
                    import_edge_index = data.edge_index[:, import_mask]
                    import_edge_type = data.edge_type[import_mask]
                    train_edge_index = torch.cat([train_edge_index, import_edge_index], dim=1)
                    train_edge_type = torch.cat([train_edge_type, import_edge_type], dim=0)

                # 初始化分数列表
                precision_scores = []
                recall_scores = []
                f1_scores = []

                for i in range(50):
                    print(
                        f"Dataset: {dataset}, Node Feature: {nodes_feature}, Edge Types: {edge_types}, Experiment {i + 1}/50")
                    transform = T.RandomLinkSplit(
                        num_test=0.75,
                        disjoint_train_ratio=0.3,
                        neg_sampling_ratio=2.0,
                        is_undirected=True,
                        add_negative_train_samples=False
                    )
                    train_link, _, _ = transform(link_data)
                    train_data = train_link
                    # 创建训练集Data对象
                    train_data = Data(x=train_data.x, edge_index=train_edge_index, edge_type=train_edge_type)
                    train_data = T.ToUndirected()(train_data)

                    # 使用 LinkNeighborLoader 进行采样
                    train_loader = LinkNeighborLoader(
                        data=train_data,
                        num_neighbors=[20, 10],
                        neg_sampling_ratio=2.0,
                        edge_label_index=train_link.edge_label_index,
                        edge_label=train_link.edge_label,
                        batch_size=128,
                        shuffle=True,
                    )

                    # 定义模型
                    model = RGCN_LP(in_channels=x.size(1), hidden_channels=64, num_relations=num_relations)
                    device_model = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device_model)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                    # 训练模型
                    model.train()
                    for epoch in range(1, 30):
                        total_loss = 0
                        total_examples = 0
                        for sampled_data in train_loader:
                            optimizer.zero_grad()
                            sampled_data = sampled_data.to(device_model)
                            pred = model(
                                sampled_data.x,
                                sampled_data.edge_index,
                                sampled_data.edge_type,
                                sampled_data.edge_label_index
                            )
                            ground_truth = sampled_data.edge_label.float().unsqueeze(1)
                            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item() * pred.size(0)
                            total_examples += pred.size(0)

                    # 评估模型
                    model.eval()
                    with torch.no_grad():
                        # 测试集只包含“link”边
                        test_loader = LinkNeighborLoader(
                            data=test_data,
                            num_neighbors=[20, 10],
                            edge_label_index=test_data.edge_label_index,
                            edge_label=test_data.edge_label,
                            batch_size=3 * 128,
                            shuffle=False,
                        )

                        preds = []
                        ground_truths = []
                        for sampled_data in test_loader:
                            sampled_data = sampled_data.to(device_model)
                            pred = model(
                                sampled_data.x,
                                sampled_data.edge_index,
                                sampled_data.edge_type,
                                sampled_data.edge_label_index
                            )
                            preds.append(pred.cpu())
                            ground_truths.append(sampled_data.edge_label.cpu())
                        pred = torch.cat(preds, dim=0).numpy()
                        ground_truth = torch.cat(ground_truths, dim=0).numpy()
                        pred_probs = torch.sigmoid(torch.from_numpy(pred)).squeeze(1).numpy()
                        pred_labels = (pred_probs > 0.5).astype(np.float32)

                        # 计算评估指标
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
                    'F1': avg_f1,
                })

    # 将所有结果写入Excel
    results_df = pd.DataFrame(all_results)
    os.makedirs('./new_result', exist_ok=True)
    results_df.to_excel('./result/r_gcn_rq1.xlsx', index=False)
    print("All experiments completed. Results saved to './result/r_gcn_rq1.xlsx'.")

if __name__ == '__main__':
    main()
