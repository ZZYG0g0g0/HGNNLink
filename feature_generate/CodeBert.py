import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd

if __name__ == '__main__':
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = ['Derby', 'Drools', 'Infinispan', 'iTrust', 'maven', 'Pig', 'Seam2']
    for dataset in datasets:
        # 初始化 codeBERT 模型和 tokenizer
        model_name = 'microsoft/codebert-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name)
        model.to(device)  # 将模型移动到 GPU
        model.eval()  # 设置为评估模式

        # 读取文本文件
        input_file = f'../docs/{dataset}/cc/cc_doc.txt'  # 将其替换为你的文本文件路径
        with open(input_file, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()

        # 生成 codeBERT 向量
        vectors = []
        for line in lines:
            inputs = tokenizer(line.strip(), return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入移动到 GPU
            with torch.no_grad():
                outputs = model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 取平均值作为文本的向量表示，并移动到 CPU
            vectors.append(vector)

        # 将向量写入 Excel 文件
        df = pd.DataFrame(vectors)
        output_file = f'../docs/{dataset}/cc/cc_code_bert_vectors.xlsx'  # 输出的 Excel 文件路径
        df.to_excel(output_file, index=False)

        print(f"codeBERT vectors have been written to {output_file}")
