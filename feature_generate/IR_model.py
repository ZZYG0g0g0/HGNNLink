# -*-coding:ISO8859-1-*-
# @Time : 2022/10/10 20:44
# @Author : 邓洋、李幸阜
import os

import scipy
from gensim import corpora, similarities
from gensim import models
import re
import numpy as np
import pandas as pd
import scipy.stats

from rank_bm25 import BM25Okapi

import warnings

warnings.filterwarnings(action='ignore')

# 生成查询集或被查询集(生成数据集)
def set_generation(query_file):
    """
    生成查询集或被查询集(生成数据集)
    :param query_file: 分词和去除停用词后的数据集
    :return: 返回一个列表，列表的每个元素也是一个列表，后者中的列表的每个元素都是每一条数据中的单词。
    """
    with open(query_file, "r", encoding="ISO8859-1") as ft:
        lines_T = ft.readlines()
    setline = []
    for line in lines_T:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        word = [i for i in word if len(i) > 0]
        setline.append(word)
    return setline


# VSM相似度计算
def vsm_similarity(queried_file, query_file, output_fname=None):
    # 生成被查询集
    queried_line = set_generation(queried_file)
    # 生成查询集
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_tfidf)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    sim = pd.DataFrame(corpus_sim[query_tfidf])
    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim


# LSI相似度计算
def lsi_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 生成lsi主题
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary)
    corpus_lsi = lsi_model[corpus_tfidf]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_lsi)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    query_lsi = lsi_model[query_tfidf]
    sim = pd.DataFrame(corpus_sim[query_lsi])

    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim


# LDA相似度计算
def lda_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    # tfidf_model = models.TfidfModel(corpus)
    # corpus_tfidf = tfidf_model[corpus]

    # 生成lda主题
    topic_number = 100
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_number, random_state=0)
    document_topic = lda_model.get_document_topics(corpus)
    # corpus_lda = lda_model[corpus_tfidf]

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    # query_tfidf = tfidf_model[query_corpus]
    query_lda = lda_model.get_document_topics(query_corpus)

    sim = hellingerSim(document_topic, query_lda, topic_number)

    if output_fname is not None:
        sim.to_excel(output_fname)
    print(sim)
    return sim


# BM25分数计算
def bm25_score(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    bm25 = BM25Okapi(queried_line)
    scores = pd.DataFrame(bm25.get_full_scores(query_line))
    if output_fname is not None:
        scores.to_excel(output_fname)
    return scores


# JS散度相似度计算
def JS_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line + query_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]
    corpus2 = [dictionary.doc2bow(text) for text in query_line]
    A_matrix = np.zeros((len(queried_line), len(dictionary)))
    B_matrix = np.zeros((len(query_line), len(dictionary)))

    row = 0
    for document in corpus:
        for word_id, freq in document:
            A_matrix[row][word_id] = freq
        row = row + 1

    row = 0
    for document in corpus2:
        for word_id, freq in document:
            B_matrix[row][word_id] = freq
        row = row + 1

    sum_matrix = np.sum(np.vstack((A_matrix, B_matrix)), axis=0)
    probability_A = A_matrix / sum_matrix
    probability_B = B_matrix / sum_matrix

    sim = JS_Sim(probability_A, probability_B)

    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim


def JS_Sim(A_set, B_set) -> pd.DataFrame:
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    # 开始计算JS相似度
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = JS_divergence(B_set[row], A_set[column])  # B_set为查询集，所以放前面
    return df


def JS_divergence(p, q):
    M = (p + q) / 2
    pk = np.asarray(p)
    pk2 = np.asarray(q)
    a = 0
    b = 0
    if (np.sum(pk, axis=0, keepdims=True) != 0):
        a = 0.5 * scipy.stats.entropy(p, M)
    if (np.sum(pk2, axis=0, keepdims=True) != 0):
        b = 0.5 * scipy.stats.entropy(q, M)

    return a + b  # 选用自然对数


def hellingerSim(A_set, B_set, topic_number):
    """
    计算两个集合中每条数据之间的Hellinger距离
    :param A_set: 被查询集
    :param B_set: 查询集
    :return: 一个 len(B_set) * len(A_set) 的 pandas.DataFrame
    """
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    A_matrix = np.zeros((len(A_set), topic_number))
    B_matrix = np.zeros((len(B_set), topic_number))

    # 将A_set和B_set分别转化为List[List[float]](e.i. 二维矩阵)
    row = 0
    for tu in A_set:
        for i in tu:
            A_matrix[row][i[0]] = i[1]
        row = row + 1
    row = 0
    for tu in B_set:
        for i in tu:
            B_matrix[row][i[0]] = i[1]
        row = row + 1

    # 开始计算Hellinger距离
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = HellingerDistance(B_matrix[row], A_matrix[column])  # B_matrix为查询集，所以放前面
    return df


def HellingerDistance(p, q):
    """
    计算HellingerDistance距离
    :param p:
    :param q:
    :return: float距离
    """
    return 1 - (1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q)))


def IR_based_feature_generation(fname, tname, dataset_name, output_fname=None):
    """
    生成两个制品之间链接向量(笛卡尔积个)
    :param fname: 制品1
    :param tname: 制品2
    :return: 链接向量，不带label
    """
    IR_based_feature = pd.DataFrame()
    options = [vsm_similarity, lsi_similarity, lda_similarity, bm25_score, JS_similarity]
    options_name = ['vsm', 'lsi', 'lda', 'bm25', 'JS']
    colomn_name = []
    flag = 0
    index = 0
    source_artifacts = []
    target_artifacts = []
    uc_names = os.listdir(f'../dataset/{dataset_name}/uc/')
    cc_names = os.listdir(f'../dataset/{dataset_name}/cc/')
    for i in range(len(uc_names)):
        for j in range(len(cc_names)):
            target_artifacts.append(cc_names[j].split('.')[0])
            source_artifacts.append(uc_names[i])
    colomn_name.append('requirement')
    colomn_name.append('code')
    IR_based_feature['requirement'] = source_artifacts
    IR_based_feature['code'] = target_artifacts
    for option in options:
        sim = option(fname , tname)  # tname为查询集，fname为被查询集
        IR_based_feature[flag] = pd.concat([sim.iloc[i] for i in range(sim.shape[0])], axis=0,
                                           ignore_index=True)  # 将sim所有行转化为IR_based_feature的一列

        colomn_name.append(options_name[index] + '_1')
        flag = flag + 1

        sim = option(tname, fname)  # 查询集和被查询集交换
        IR_based_feature[flag] = pd.concat([sim.iloc[:, i] for i in range(sim.shape[1])], axis=0,
                                           ignore_index=True)  # 将sim所有列转化为IR_based_feature的一列
        colomn_name.append(options_name[index] + "_2")
        flag = flag + 1
        index += 1

    if output_fname is not None:
        IR_based_feature.columns = colomn_name
        IR_based_feature.to_excel(output_fname + ".xlsx")
    return IR_based_feature

if __name__ == '__main__':
    datasets = ['iTrust']
    # datasets = ['Derby', 'Drools', 'Infinispan', 'iTrust', 'maven', 'Pig', 'Seam2']
    for dataset in datasets:
        print(dataset)
        fname = "../docs/"+dataset+"/cc/cc_doc.txt"
        tname = "../docs/"+dataset+"/uc/uc_doc.txt"
        output_fname = "../docs/"+dataset+"/IR_feature"
        IR_based_feature_generation(fname, tname, dataset, output_fname)