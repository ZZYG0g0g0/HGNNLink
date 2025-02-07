import pandas as pd
import os

#获取sigma
def get_sigma(dataset_name, ir_model_name):
    # 读取相似度数据
    data = pd.read_excel(f'../docs/{ir_model_name}/{dataset_name}.xlsx')

    # 计算sigma
    grouped = data.groupby('requirement')['similarity']
    max_values = grouped.max()
    min_values = grouped.min()

    # 计算 vi
    v_i = (max_values - min_values) / 2

    # 计算 δ (sigma)
    sigma = v_i.median()

    return sigma

def get_sorted_df(dataset_name, ir_model_name):
    # 读取相似度数据
    data = pd.read_excel(f'../docs/{ir_model_name}/{dataset_name}.xlsx')

    return data.sort_values(by='similarity', ascending=False)

def get_uc_cc_dict(dataset_name):
    # 获取需求和代码的文件名列表
    uc_names = os.listdir(f'../dataset/{dataset_name}/uc')
    cc_names = os.listdir(f'../dataset/{dataset_name}/cc')

    # 创建索引字典
    uc_idx_dict = {uc_names[i]: i for i in range(len(uc_names))}
    cc_idx_dict = {cc_names[j]: j for j in range(len(cc_names))}

    return uc_idx_dict, cc_idx_dict


def generate_source_target_edge_uc_cc(dataset_name, edge_percent, sorted_data, uc_idx_dict, cc_idx_dict):
    # 计算边界索引，根据 edge_percent 获取前 x% 的数据
    num_edges = len(sorted_data)
    cutoff_index = int(num_edges * edge_percent)

    # 取前 edge_percent 的数据和后 edge_percent 的数据
    top_links = sorted_data.iloc[:cutoff_index]
    if cutoff_index == 0:
        last_links = sorted_data.iloc[0:0]  # 返回一个空的DataFrame
    else:
        last_links = sorted_data.iloc[-cutoff_index:]

    middle_links = sorted_data.iloc[cutoff_index:-cutoff_index]

    # 使用向量化方法生成边关系
    top_edge_from = top_links['requirement'].map(uc_idx_dict).values
    top_edge_to = top_links['code'].map(cc_idx_dict).values

    negative_edge_from = last_links['requirement'].map(uc_idx_dict).values
    negative_edge_to = last_links['code'].map(cc_idx_dict).values

    test_edge_from = middle_links['requirement'].map(uc_idx_dict).values
    test_edge_to = middle_links['code'].map(cc_idx_dict).values

    # 生成测试集标签
    test_data_label = generate_test_data_label(middle_links, dataset_name)
    top_data_label = generate_test_data_label(top_links, dataset_name)
    last_data_label = generate_test_data_label(last_links, dataset_name)

    return top_edge_from, top_edge_to, negative_edge_from, negative_edge_to, test_edge_from, test_edge_to, test_data_label, top_data_label, last_data_label, middle_links, top_links, last_links

def generate_test_data_label(original_df, dataset):
    # 打开文件，并指定编码
    with open(f'../dataset/{dataset}/true_set.txt', 'r', encoding='ISO8859-1') as tf:
        # 将真值集转换为集合以加快搜索速度
        ground_truth = set(line.strip() for line in tf)

    # 假设 links 是从另一个 DataFrame 获得的
    links = original_df.copy()

    # 先创建一个新列，存储每行构造的字符串
    links['link_str'] = links['requirement'] + ' ' + links['code']

    # 然后应用集合检查
    links['label'] = links['link_str'].apply(lambda x: 1.0 if x in ground_truth else 0.0)

    # 返回结果列
    return links['label'].tolist()

#计算精确率和召回率
def calculate_precision_recall(df, percentile, label_col):
    # 计算截止索引
    cutoff_index = int(len(df) * percentile)

    # 获取预测为正例的子集
    predicted_positives = df.iloc[:cutoff_index]

    # 真实正例的数量
    actual_positives = df[df[label_col] == 1]

    # 真正例（True Positives）的数量：被预测为正例且实际为正例
    true_positives = predicted_positives[predicted_positives[label_col] == 1]

    # 计算精确率和召回率
    precision = len(true_positives) / len(predicted_positives) if len(predicted_positives) > 0 else 0
    recall = len(true_positives) / len(actual_positives) if len(actual_positives) > 0 else 0

    return precision, recall

# 定义计算AP的函数
def calculate_ap(group):
    # 按相似度降序排序
    group = group.sort_values('similarity', ascending=False)
    # 计算是否相关
    relevant = group['label'] == 1
    # 计算累计正确数
    cumsum = relevant.cumsum()
    # 计算precision@k
    precision_at_k = cumsum / (range(1, len(relevant) + 1))
    if relevant.sum() == 0:
        return 0
    # 计算AP
    ap = (precision_at_k * relevant).sum() / relevant.sum()
    return ap

#生成继承边
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
        # else:
        #     print(f"Error: {cc_name1} or {cc_name2} not found in dictionaries.")
    return edge_from, edge_to

#生成调用边
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
        # else:
        #     print(f"Error: {cc_name1} or {cc_name2} not found in dictionaries.")
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
