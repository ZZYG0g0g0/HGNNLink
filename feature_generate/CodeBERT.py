import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from tqdm import tqdm

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_texts(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',  # 使用推荐的 padding 参数
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets = ['iTrust']
    datasets = ['Groovy', 'Dronology', 'smos']
    types = ['cc']  # 只处理 cc
    max_length = 512
    batch_size = 8

    for dataset in datasets:
        for type in types:
            try:
                # 使用 CodeBERT 模型
                model_name = 'microsoft/codebert-base'
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                model = RobertaModel.from_pretrained(model_name)
                model.to(device)
                model.eval()

                input_file = f'../docs/{dataset}/{type}/{type}_emb_doc.txt'
                with open(input_file, 'r', encoding='ISO-8859-1') as f:
                    lines = f.readlines()

                input_ids, attention_masks = preprocess_texts(lines, tokenizer, max_length)
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)

                vectors = []
                with torch.no_grad():
                    for i in tqdm(range(0, len(input_ids), batch_size)):
                        batch_input_ids = input_ids[i:i+batch_size]
                        batch_attention_masks = attention_masks[i:i+batch_size]
                        outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
                        batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        vectors.extend(batch_vectors)

                df = pd.DataFrame(vectors)
                output_file = f'../docs/{dataset}/{type}/{type}_codebert_vectors.xlsx'
                ensure_dir(output_file)
                df.to_excel(output_file, index=False)

                print(f"CodeBERT vectors have been written to {output_file}")
                print(f"Sample vector shape: {vectors[0].shape}")
            except Exception as e:
                print(f"Error processing {dataset}/{type}: {e}")
