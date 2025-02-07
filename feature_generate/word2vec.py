import gensim
import pandas as pd
import numpy as np

# 读取文本文件
def read_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        texts = file.readlines()
    texts = [text.strip() for text in texts]
    return texts

# 加载预训练的 Word2Vec 模型
def load_pretrained_word2vec_model(model_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model

# 获取每个文本的向量
def get_text_vectors(model, texts):
    vectors = []
    for text in texts:
        words = text.split()
        word_vectors = [model[word] for word in words if word in model]
        if word_vectors:
            vector = np.mean(word_vectors, axis=0)
        else:
            vector = np.zeros(model.vector_size)
        vectors.append(vector)
    return vectors

# 将向量写入 Excel 文件
def write_vectors_to_excel(vectors, file_path):
    df = pd.DataFrame(vectors)
    df.to_excel(file_path, index=False)

# 主函数
def main():
    datasets = ['iTrust']
    # datasets = ['Derby', 'Drools', 'Infinispan', 'iTrust', 'maven', 'Pig', 'Seam2']
    # types = ['uc', 'cc']
    types = ['uc']
    for dataset in datasets:
        for type in types:
            # 文件路径
            input_file_path = '../docs/' + dataset + '/'+type+'/'+type+'_doc.txt'
            output_file_path = '../docs/' + dataset + '/'+type+'/'+type+'_word2vec_vectors.xlsx'
            pretrained_model_path = '../GoogleNews-vectors-negative300.bin'

            # 读取和预处理文本
            texts = read_texts(input_file_path)

            # 加载预训练的 Word2Vec 模型
            model = load_pretrained_word2vec_model(pretrained_model_path)

            # 获取每个文本的向量
            vectors = get_text_vectors(model, texts)

            # 将向量写入 Excel 文件
            write_vectors_to_excel(vectors, output_file_path)
            print(f'Word2Vec vectors have been written to {output_file_path}')

if __name__ == '__main__':
    main()
