from utils import get_uc_cc_dict
from utils import generate_source_target_edge_uc_cc
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import torch.nn.functional as F
from utils import get_sigma,generate_extend_edges, generate_import_edges, generate_IR_edges
from utils import calculate_precision_recall
from utils import get_sorted_df
from HGT import Model as HGT_Model
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def hgt(datasets, node_features, ir_models, output_file_path, metric):
    with pd.ExcelWriter(output_file_path) as writer:
        for dataset in datasets:
            results = []
            uc_dict, cc_dict = get_uc_cc_dict(dataset)
            for node_feature in node_features:
                print(f'{dataset}->{node_feature}')
                # 获取uc和cc的结点特征
                uc_df = pd.read_excel(f'../docs/{dataset}/uc/uc_{node_feature}_vectors.xlsx')
                cc_df = pd.read_excel(f'../docs/{dataset}/cc/cc_{node_feature}_vectors.xlsx')
                # 转换成torch.float格式
                req_feat = torch.from_numpy(uc_df.values).to(torch.float)
                code_feat = torch.from_numpy(cc_df.values).to(torch.float)
                # 第一部分：从0.001到0.011，步长为0.001
                part1 = np.arange(0.001, 0.01, 0.001)
                # 第二部分：从0.01到0.51，步长为0.1
                part2 = np.arange(0.01, 0.15, 0.01)
                # 将两部分连接起来
                thresholds = np.concatenate((part1, part2))
                for ir_model in ir_models:
                    sorted_data = get_sorted_df(dataset, ir_model)
                    sigma = get_sigma(dataset, ir_model)
                    for threshold1 in thresholds:
                        print(threshold1)
                        if threshold1 * sorted_data.shape[0] < 1:
                            part1 = np.arange(0, 0.01, 0.001)
                            part2 = np.arange(0.01, 1.01, 0.01)
                            thresholds_2 = np.concatenate((part1, part2))
                            for threshold_2 in thresholds_2:
                                # 收集数据
                                results.append({
                                    'Dataset': dataset,
                                    'Threshold1': threshold1,
                                    'Threshold2': threshold_2,
                                    'Precision': 0,
                                    'Recall': 0,
                                    'F1': 0
                                })
                            continue

                        # 1) 生成训练/测试集主边
                        (edge_from, edge_to,
                         neg_from, neg_to,
                         test_from, test_to,
                         test_label, top_label, last_label,
                         middle_links, top_links, last_links
                         ) = generate_source_target_edge_uc_cc(dataset, threshold1, sorted_data, uc_dict, cc_dict)

                        pos_edges = torch.tensor(np.array([edge_from, edge_to]), dtype=torch.long)
                        neg_edges = torch.tensor(np.array([neg_from, neg_to]), dtype=torch.long)
                        test_edges = torch.tensor(np.array([test_from, test_to]), dtype=torch.long)
                        pos_labels = torch.ones(pos_edges.size(1), dtype=torch.float32)
                        neg_labels = torch.zeros(neg_edges.size(1), dtype=torch.float32)
                        test_set_labels = torch.tensor(test_label, dtype=torch.float32)

                        # 2) 构建带三条额外边的图
                        data_hetero = HeteroData()
                        data_hetero["req"].x = req_feat
                        data_hetero["code"].x = code_feat

                        # req->code 主边 (正+负)
                        data_hetero["req", "link", "code"].edge_index = torch.cat([pos_edges, neg_edges], dim=1)
                        data_hetero["req", "link", "code"].edge_label = torch.cat([pos_labels, neg_labels], dim=0)
                        data_hetero["req", "link", "code"].edge_label_index = data_hetero[
                            "req", "link", "code"].edge_index

                        # 读取三条关系边
                        extend_edge_from, extend_edge_to = generate_extend_edges(dataset)
                        import_edge_from, import_edge_to = generate_import_edges(dataset)
                        IR_edge_from, IR_edge_to = generate_IR_edges(dataset)

                        # 添加 extend 边 (code->code)
                        if len(extend_edge_from) > 0:
                            data_hetero["code", "extend", "code"].edge_index = torch.tensor(
                                [extend_edge_from, extend_edge_to], dtype=torch.long
                            )

                        # 添加 import 边 (code->code)
                        if len(import_edge_from) > 0:
                            data_hetero["code", "import", "code"].edge_index = torch.tensor(
                                [import_edge_from, import_edge_to], dtype=torch.long
                            )

                        # 添加 IR 边 (req->code), 这里假设关系命名为 "sim"
                        if len(IR_edge_from) > 0:
                            data_hetero["req", "sim", "code"].edge_index = torch.tensor(
                                [IR_edge_from, IR_edge_to], dtype=torch.long
                            )

                        data_hetero = T.ToUndirected()(data_hetero)

                        # 3) 测试图
                        test_data_hetero = HeteroData()
                        test_data_hetero["req"].x = req_feat
                        test_data_hetero["code"].x = code_feat
                        test_data_hetero["req", "link", "code"].edge_index = test_edges
                        test_data_hetero["req", "link", "code"].edge_label = test_set_labels
                        test_data_hetero["req", "link", "code"].edge_label_index = test_data_hetero[
                            "req", "link", "code"].edge_index

                        # 同样加上 extend, import, IR 到测试图
                        if len(extend_edge_from) > 0:
                            test_data_hetero["code", "extend", "code"].edge_index = torch.tensor(
                                [extend_edge_from, extend_edge_to], dtype=torch.long
                            )
                        if len(import_edge_from) > 0:
                            test_data_hetero["code", "import", "code"].edge_index = torch.tensor(
                                [import_edge_from, import_edge_to], dtype=torch.long
                            )
                        if len(IR_edge_from) > 0:
                            test_data_hetero["req", "sim", "code"].edge_index = torch.tensor(
                                [IR_edge_from, IR_edge_to], dtype=torch.long
                            )

                        test_data_hetero = T.ToUndirected()(test_data_hetero)
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = HGT_Model(in_channels=req_feat.size(1), out_channels=128,
                                         metadata=data_hetero.metadata()).to(device)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

                        train_loader = LinkNeighborLoader(
                            data=data_hetero,
                            num_neighbors=[20, 10],
                            edge_label_index=(
                            ("req", "link", "code"), data_hetero["req", "link", "code"].edge_label_index),
                            edge_label=data_hetero["req", "link", "code"].edge_label,
                            batch_size=128,
                            shuffle=True,
                        )
                        test_loader = LinkNeighborLoader(
                            data=test_data_hetero,
                            num_neighbors=[20, 10],
                            edge_label_index=(
                            ("req", "link", "code"), test_data_hetero["req", "link", "code"].edge_label_index),
                            edge_label=test_data_hetero["req", "link", "code"].edge_label,
                            batch_size=3 * 128,
                            shuffle=False,
                        )
                        # 训练
                        for _ep in range(30):
                            for batch_data in train_loader:
                                batch_data = batch_data.to(device)
                                optimizer.zero_grad()
                                out = model(batch_data)
                                loss = F.binary_cross_entropy_with_logits(
                                    out, batch_data["req", "link", "code"].edge_label.float()
                                )
                                loss.backward()
                                optimizer.step()

                        # 5) 测试阶段
                        middle_links_copy = middle_links.copy()
                        with torch.no_grad():
                            idx = 0
                            for batch_test in test_loader:
                                batch_test = batch_test.to(device)
                                logits = model(batch_test)
                                preds = (logits > 0.5).float().cpu().numpy()
                                for lab in preds:
                                    if lab == 1:
                                        # 根据 GA 论文中介绍的方式累加 similarity
                                        middle_links_copy.iloc[
                                            idx, middle_links_copy.columns.get_loc('similarity')] += \
                                            sigma * middle_links.iloc[
                                                idx, middle_links.columns.get_loc('similarity')]
                                    idx += 1

                            middle_links_copy.sort_values('similarity', inplace=True, ascending=False)
                            label_all = top_label + test_label + last_label
                            # 对 top_links, last_links 做极大/极小偏移
                            top_links.loc[:, 'similarity'] += 999
                            last_links.loc[:, 'similarity'] -= 999

                            mid_merged = middle_links.merge(
                                middle_links_copy, on=['requirement', 'code'], suffixes=('', '_copy')
                            )
                            mid_merged.drop('similarity', axis=1, inplace=True)
                            mid_merged.rename(columns={'similarity_copy': 'similarity'}, inplace=True)

                            results_df = pd.concat([top_links, mid_merged, last_links], ignore_index=True)
                            results_df['label'] = label_all

                            # 如果要计算P-R值
                            if metric == 'PR':
                                # 按照similarity排序
                                sorted_df = results_df.sort_values(by='similarity', ascending=False)
                                part1 = np.arange(0, 0.01, 0.001)
                                part2 = np.arange(0.01, 1.01, 0.01)
                                thresholds_2 = np.concatenate((part1, part2))
                                for threshold_2 in thresholds_2:
                                    # print(threshold_2)
                                    precision, recall = calculate_precision_recall(sorted_df, threshold_2,
                                                                                   'label')
                                    if precision + recall == 0:
                                        f1 = 0
                                    else:
                                        f1 = 2 * precision * recall / (precision + recall)
                                    results.append({
                                        'Dataset': dataset,
                                        'Node_feature': node_feature,
                                        'Ir_model': ir_model,
                                        'Threshold1': threshold1,
                                        'Threshold2': threshold_2,
                                        'Precision': precision,
                                        'Recall': recall,
                                        'F1': f1
                                    })
                        excel_df = pd.DataFrame(results)
                        excel_df.to_excel(writer, sheet_name=dataset, index=False)



if __name__ == '__main__':
    datasets = ['Seam2','iTrust', 'maven', 'Pig', 'Infinispan', 'Derby', 'Drools']
    node_features = ['albert']
    ir_models = ["IR_best"]
    output_file = f'./result/No_GA.xlsx'
    metric = 'PR'
    hgt(datasets, node_features, ir_models, output_file, metric)