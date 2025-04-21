import os
import torch
from transformers import RobertaTokenizer, T5EncoderModel
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
        if not isinstance(text, str):
            text = str(text)
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
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

    datasets = ['Derby']
    types = ['cc']
    max_length = 512
    batch_size = 8

    for dataset in datasets:
        for type in types:
            try:
                # 关键修改：使用 RobertaTokenizer
                model_name = 'Salesforce/codet5-base'
                tokenizer = RobertaTokenizer.from_pretrained(model_name)  # 正确分词器
                model = T5EncoderModel.from_pretrained(model_name)         # 根据任务选择模型
                model.to(device)
                model.eval()

                input_file = f'../docs/{dataset}/{type}/{type}_emb_doc.txt'
                with open(input_file, 'r', encoding='ISO-8859-1') as f:
                    lines = [str(line).strip() for line in f if line.strip()]  # 强制转字符串

                input_ids, attention_masks = preprocess_texts(lines, tokenizer, max_length)
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)

                vectors = []
                with torch.no_grad():
                    for i in tqdm(range(0, len(input_ids), batch_size)):
                        batch_input_ids = input_ids[i:i+batch_size]
                        batch_attention_masks = attention_masks[i:i+batch_size]
                        outputs = model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_masks
                        )
                        batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        vectors.extend(batch_vectors)

                df = pd.DataFrame(vectors)
                output_file = f'../docs/{dataset}/{type}/{type}_codet5_vectors.xlsx'
                ensure_dir(output_file)
                df.to_excel(output_file, index=False)

                print(f"CodeT5 vectors written to {output_file}")
                if vectors:
                    print(f"Sample vector shape: {vectors[0].shape}")

            except Exception as e:
                print(f"Error processing {dataset}/{type}: {e}")