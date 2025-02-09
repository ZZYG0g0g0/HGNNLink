import pandas as pd

# 定义读取候选链接和真集的方法
def read_candidate_links(dataset):
    filepath = f"../docs/IR_best/{dataset}.xlsx"
    return pd.read_excel(filepath)


def read_true_set(dataset):
    filepath = f"../dataset/{dataset}/true_set.txt"
    with open(filepath, 'r') as file:
        true_links = set(line.strip() for line in file)
    return true_links


# 定义计算F1值、精确率和召回率的方法
def calculate_metrics(candidate_links, true_links, threshold):
    num_links_to_select = int(len(candidate_links) * threshold)
    selected_links = set(
        f"{row['requirement']} {row['code']}" for _, row in candidate_links.iloc[:num_links_to_select].iterrows()
    )

    true_positive = selected_links & true_links
    false_positive = selected_links - true_links
    false_negative = true_links - selected_links

    precision = len(true_positive) / (len(true_positive) + len(false_positive)) if len(selected_links) > 0 else 0
    recall = len(true_positive) / (len(true_positive) + len(false_negative)) if len(true_links) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


# 主逻辑
def process_datasets(datasets):
    writer = pd.ExcelWriter('../RQ3/result/IR_result.xlsx', engine='openpyxl')

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        candidate_links = read_candidate_links(dataset)
        true_links = read_true_set(dataset)

        results = []
        for threshold in [i / 100 for i in range(1, 101)]:
            precision, recall, f1 = calculate_metrics(candidate_links, true_links, threshold)
            results.append({
                'Dataset': dataset,
                'Threshold': threshold,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })

        results_df = pd.DataFrame(results)
        results_df.to_excel(writer, sheet_name=dataset, index=False)

    writer.close()
    print("Results saved to IR_result.xlsx")

if __name__ == '__main__':
    # 数据集列表
    datasets = ["iTrust", "maven", "Seam2", "Infinispan", "Derby", "Drools", "Pig"]  # 替换为实际数据集名称
    process_datasets(datasets)
