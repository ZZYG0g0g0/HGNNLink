import numpy as np
import random
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from deap import base, creator, tools
import pandas as pd
from tqdm import tqdm
import warnings
from utils import (
    get_sorted_df, get_sigma, get_uc_cc_dict, generate_source_target_edge_uc_cc,
    generate_extend_edges, generate_import_edges, generate_IR_edges,
    calculate_precision_recall
)

from HGT import Model as HGTModel

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def custom_ea_simple(population, toolbox, cxpb, mutpb, ngen):
    """
    自定义的 eaSimple，增加了进度条 & 每5代打印一次所有个体及其fitness
    """
    logbook = tools.Logbook()

    # 评估初始群体
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 用 tqdm 做进度条
    for gen in tqdm(range(1, ngen + 1), desc="GA Searching"):
        # 1) 选择
        offspring = toolbox.select(population, len(population))
        # 2) 克隆
        offspring = list(map(toolbox.clone, offspring))

        # 3) 交叉
        for child1, child2 in zip(offspring[0::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                # 交叉后fitness失效，需要重算
                del child1.fitness.values, child2.fitness.values

        # 4) 变异
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                # 变异后fitness失效，需要重算
                del mutant.fitness.values

        # 5) 评估无效适应度
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind_, fit_ in zip(invalid_ind, fitnesses):
            ind_.fitness.values = fit_

        # 6) 用新后代替换旧种群
        population[:] = offspring

        # ---- 每 5 代打印一次所有个体的参数和fitness ----
        if gen % 5 == 0:
            print(f"=== Generation {gen} ===")
            for i, ind_ in enumerate(population):
                thr1 = ind_[0]
                thr2 = ind_[1]
                f_val = ind_.fitness.values[0]
                print(f"  Ind_{i}: threshold1={thr1:.4f}, threshold2={thr2:.4f}, fitness(F1)={f_val:.4f}")

    return population, logbook


def best_param_search(dataset, ir_model):
    """
    使用GA搜索最优threshold1, threshold2，节点特征固定为 'albert'，
    并在构图时额外添加 extend、import、IR 三条边。
    """
    # 读取并缓存公共数据
    sorted_data = get_sorted_df(dataset, ir_model)
    sigma = get_sigma(dataset, ir_model)
    uc_dict, cc_dict = get_uc_cc_dict(dataset)

    # GA：单目标最大化(F1)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # 仅搜索 threshold1 ∈ [0.001, 0.15], threshold2 ∈ [0.0, 1.0]
    BOUND_LOW = [0.001, 0.0]
    BOUND_UP = [0.15, 1.0]

    def init_ind(icls, low, up):
        threshold1 = random.uniform(low[0], up[0])
        threshold2 = random.uniform(low[1], up[1])
        return icls([threshold1, threshold2])

    toolbox.register("individual", init_ind, creator.Individual, BOUND_LOW, BOUND_UP)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # GA 算子
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.05, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_individual(ind):
        threshold1 = ind[0]
        threshold2 = ind[1]
        node_feature = "albert"  # 固定为 albert

        # 若 threshold1 太小以致 top 集为空，则给极低 fitness 惩罚
        if threshold1 * sorted_data.shape[0] < 1:
            return (0.0,)

        # 读取该 node_feature 的向量
        uc_df = pd.read_excel(f'../docs/{dataset}/uc/uc_{node_feature}_vectors.xlsx')
        cc_df = pd.read_excel(f'../docs/{dataset}/cc/cc_{node_feature}_vectors.xlsx')
        req_feat = torch.from_numpy(uc_df.values).to(torch.float)
        code_feat = torch.from_numpy(cc_df.values).to(torch.float)

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
        data_hetero["req", "link", "code"].edge_label_index = data_hetero["req", "link", "code"].edge_index

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
        test_data_hetero["req", "link", "code"].edge_label_index = test_data_hetero["req", "link", "code"].edge_index

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

        # 4) 初始化模型并训练
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HGTModel(in_channels=req_feat.size(1), out_channels=128, metadata=data_hetero.metadata()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        train_loader = LinkNeighborLoader(
            data=data_hetero,
            num_neighbors=[20, 10],
            edge_label_index=(("req", "link", "code"), data_hetero["req", "link", "code"].edge_label_index),
            edge_label=data_hetero["req", "link", "code"].edge_label,
            batch_size=128,
            shuffle=True,
        )
        test_loader = LinkNeighborLoader(
            data=test_data_hetero,
            num_neighbors=[20, 10],
            edge_label_index=(("req", "link", "code"), test_data_hetero["req", "link", "code"].edge_label_index),
            edge_label=test_data_hetero["req", "link", "code"].edge_label,
            batch_size=3 * 128,
            shuffle=False,
        )

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
                        middle_links_copy.iloc[idx, middle_links_copy.columns.get_loc('similarity')] += \
                            sigma * middle_links.iloc[idx, middle_links.columns.get_loc('similarity')]
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

            # 计算 P, R, F
            sorted_df_2 = results_df.sort_values(by='similarity', ascending=False)
            precision_val, recall_val = calculate_precision_recall(sorted_df_2, threshold2, 'label')
            if precision_val + recall_val == 0:
                f_val = 0.0
            else:
                f_val = 2 * precision_val * recall_val / (precision_val + recall_val)

        return (f_val,)

    # 注册
    toolbox.register("evaluate", evaluate_individual)

    # 运行 GA
    pop_size = 50
    NGEN = 200
    CXPB = 0.9
    MUTPB = 0.1

    pop = toolbox.population(n=pop_size)
    pop, _ = custom_ea_simple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN)

    # 取得最优个体
    best_ind = tools.selBest(pop, k=1)[0]
    best_threshold1 = best_ind[0]
    best_threshold2 = best_ind[1]
    print(f"[best_param_search] Best => threshold1={best_threshold1:.4f}, "
          f"threshold2={best_threshold2:.4f}, node_feature=albert")
    return best_threshold1, best_threshold2, "albert"


def best_param(dataset, ir_model, best_threshold1, best_threshold2, runs=50):
    """
    用给定的 threshold1, threshold2（节点特征固定='albert'）跑多次，返回平均P/R/F。
    这里也要添加 extend/import/IR 三条边。
    """
    sorted_data = get_sorted_df(dataset, ir_model)
    sigma = get_sigma(dataset, ir_model)
    if best_threshold1 * sorted_data.shape[0] < 1:
        print("[test_best_param] best_threshold1太小，直接返回0")
        return 0.0, 0.0, 0.0

    # 固定节点特征 => 'albert'
    node_feature = "albert"
    uc_dict, cc_dict = get_uc_cc_dict(dataset)

    uc_df = pd.read_excel(f'../docs/{dataset}/uc/uc_{node_feature}_vectors.xlsx')
    cc_df = pd.read_excel(f'../docs/{dataset}/cc/cc_{node_feature}_vectors.xlsx')
    req_feat = torch.from_numpy(uc_df.values).to(torch.float)
    code_feat = torch.from_numpy(cc_df.values).to(torch.float)

    p_list, r_list, f_list = [], [], []

    for i in range(runs):
        print(i)
        # 1) 生成主边
        (edge_from, edge_to,
         neg_from, neg_to,
         test_from, test_to,
         test_label, top_label, last_label,
         middle_links, top_links, last_links
         ) = generate_source_target_edge_uc_cc(dataset, best_threshold1, sorted_data, uc_dict, cc_dict)

        pos_edges = torch.tensor(np.array([edge_from, edge_to]), dtype=torch.long)
        neg_edges = torch.tensor(np.array([neg_from, neg_to]), dtype=torch.long)
        test_edges = torch.tensor(np.array([test_from, test_to]), dtype=torch.long)
        pos_labels = torch.ones(pos_edges.size(1), dtype=torch.float32)
        neg_labels = torch.zeros(neg_edges.size(1), dtype=torch.float32)
        test_set_labels = torch.tensor(test_label, dtype=torch.float32)

        # 2) 构图 + 三条边
        data_hetero = HeteroData()
        data_hetero["req"].x = req_feat
        data_hetero["code"].x = code_feat
        data_hetero["req", "link", "code"].edge_index = torch.cat([pos_edges, neg_edges], dim=1)
        data_hetero["req", "link", "code"].edge_label = torch.cat([pos_labels, neg_labels], dim=0)
        data_hetero["req", "link", "code"].edge_label_index = data_hetero["req", "link", "code"].edge_index

        # 读取 extend/import/IR 边
        extend_edge_from, extend_edge_to = generate_extend_edges(dataset)
        import_edge_from, import_edge_to = generate_import_edges(dataset)
        IR_edge_from, IR_edge_to = generate_IR_edges(dataset)

        if len(extend_edge_from) > 0:
            data_hetero["code", "extend", "code"].edge_index = torch.tensor(
                [extend_edge_from, extend_edge_to], dtype=torch.long
            )
        if len(import_edge_from) > 0:
            data_hetero["code", "import", "code"].edge_index = torch.tensor(
                [import_edge_from, import_edge_to], dtype=torch.long
            )
        if len(IR_edge_from) > 0:
            data_hetero["req", "sim", "code"].edge_index = torch.tensor(
                [IR_edge_from, IR_edge_to], dtype=torch.long
            )

        data_hetero = T.ToUndirected()(data_hetero)

        # 测试图
        test_data_hetero = HeteroData()
        test_data_hetero["req"].x = req_feat
        test_data_hetero["code"].x = code_feat
        test_data_hetero["req", "link", "code"].edge_index = test_edges
        test_data_hetero["req", "link", "code"].edge_label = test_set_labels
        test_data_hetero["req", "link", "code"].edge_label_index = test_data_hetero["req", "link", "code"].edge_index

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

        # 3) 训练
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HGTModel(in_channels=req_feat.size(1), out_channels=128, metadata=data_hetero.metadata()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        train_loader = LinkNeighborLoader(
            data=data_hetero,
            num_neighbors=[20, 10],
            edge_label_index=(("req", "link", "code"), data_hetero["req", "link", "code"].edge_label_index),
            edge_label=data_hetero["req", "link", "code"].edge_label,
            batch_size=128,
            shuffle=True,
        )
        test_loader = LinkNeighborLoader(
            data=test_data_hetero,
            num_neighbors=[20, 10],
            edge_label_index=(("req", "link", "code"), test_data_hetero["req", "link", "code"].edge_label_index),
            edge_label=test_data_hetero["req", "link", "code"].edge_label,
            batch_size=3 * 128,
            shuffle=False,
        )

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

        # 4) 测试
        middle_links_copy = middle_links.copy()
        with torch.no_grad():
            idx2 = 0
            for batch_test2 in test_loader:
                batch_test2 = batch_test2.to(device)
                out2 = model(batch_test2)
                preds = (out2 > 0.5).float().cpu().numpy()
                for lb in preds:
                    if lb == 1:
                        middle_links_copy.iloc[idx2, middle_links_copy.columns.get_loc('similarity')] += \
                            sigma * middle_links.iloc[idx2, middle_links.columns.get_loc('similarity')]
                    idx2 += 1

            middle_links_copy.sort_values('similarity', inplace=True, ascending=False)
            label_all = top_label + test_label + last_label
            top_links.loc[:, 'similarity'] += 999
            last_links.loc[:, 'similarity'] -= 999

            mid_merged = middle_links.merge(
                middle_links_copy, on=['requirement', 'code'], suffixes=('', '_copy')
            )
            mid_merged.drop('similarity', axis=1, inplace=True)
            mid_merged.rename(columns={'similarity_copy': 'similarity'}, inplace=True)
            results_df2 = pd.concat([top_links, mid_merged, last_links], ignore_index=True)
            results_df2['label'] = label_all

            sorted_final = results_df2.sort_values(by='similarity', ascending=False)
            precision_val, recall_val = calculate_precision_recall(sorted_final, best_threshold2, 'label')
            if precision_val + recall_val == 0:
                f_val = 0.0
            else:
                f_val = 2 * precision_val * recall_val / (precision_val + recall_val)

            p_list.append(precision_val)
            r_list.append(recall_val)
            f_list.append(f_val)

    # 求平均
    p_mean = np.mean(p_list)
    r_mean = np.mean(r_list)
    f_mean = np.mean(f_list)
    return p_mean, r_mean, f_mean


def hgt(datasets, ir_models, output_file_path):
    """
    主函数：
    1) GA 搜索最优 threshold1, threshold2（节点特征固定 albert），并添加 extend/import/IR 三条边。
    2) 用最优参数跑 50 次测试，得到平均 (P, R, F)。
    3) 将结果写入 Excel。
    """
    with pd.ExcelWriter(output_file_path) as writer:
        all_results = []

        for dataset in datasets:
            dataset_results = []

            for ir_model in ir_models:
                # 1) 搜索最佳参数
                best_threshold1, best_threshold2, best_node_feature = best_param_search(dataset, ir_model)
                # best_threshold1, best_threshold2, best_node_feature = 0.1196, 0.0813, "albert"
                # 2) 重复 50 次测 P, R, F
                p_mean, r_mean, f_mean = best_param(dataset, ir_model, best_threshold1, best_threshold2, runs=50)

                print(f"[Final 50 runs] dataset={dataset}, ir_model={ir_model}, "
                      f"node_feature={best_node_feature}, threshold1={best_threshold1:.4f}, "
                      f"threshold2={best_threshold2:.4f} => P={p_mean:.4f}, R={r_mean:.4f}, F={f_mean:.4f}")

                dataset_results.append({
                    'Dataset': dataset,
                    'Ir_model': ir_model,
                    'Node_feature': best_node_feature,
                    'Threshold1': best_threshold1,
                    'Threshold2': best_threshold2,
                    'Precision': p_mean,
                    'Recall': r_mean,
                    'F1': f_mean
                })

            # 每个 dataset 一个 sheet
            df_dataset = pd.DataFrame(dataset_results)
            df_dataset.to_excel(writer, sheet_name=dataset, index=False)
            all_results.extend(dataset_results)

    print("=== 全部完成！结果已写入 Excel：", output_file_path)


if __name__ == '__main__':
    # 例子：对单个数据集 + 单个 IR_model 搜索参数，然后写入Excel
    datasets = ['Derby']
    ir_models = ["IR_best"]
    output_file = 'result/Derby_GA.xlsx'
    hgt(datasets, ir_models, output_file)