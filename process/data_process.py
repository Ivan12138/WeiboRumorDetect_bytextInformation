#!/usr/bin/env python
# encoding:utf-8

"""
@Proofreading:wh
@file: data_process.py
@description:通过将语料库中词语的分词后，通过去掉标点后，
        与停用词表对比，将不属于停用词表的词提取出来，作为x特征，然后该语料的label作为y训练
        然后将数据传过去，predict=1时，判断为谣言。
"""

import pandas as pd
import numpy as np
import jieba
import re
import pickle

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

from sklearn.pipeline import Pipeline


# 显示所有的列
pd.set_option('display.max_columns', None)
# 显示所有的行
pd.set_option('display.max_row', None)
# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)

# 文件的绝对路径以及相对路径
# rumor_file = '../dataset/rumor.csv'
# non_rumor_file = '../dataset/non_rumor.csv'
weibo_data_file = '../dataset/weibo_data.csv'
stopwords_file = '../dataset/stopwords_ZE.txt'

#去标点符号
def remove_punctuation(line):
    """

    :param line:对每句话去标点符号
    :return: 返回处理好的每句话
    """
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u'[^a-zA-Z0-9\u4E00-\u9FA5]')
    line = rule.sub('', line)
    return line


def stopwords_list(filepath):
    """
    返回停用词表
    :param filepath:
    :return:
    """
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readline()]
    return stopwords


def processTextFeature(csv_file):
    """
    特征处理
    :param csv_file:
    :return:
    """
    rumor_df = pd.read_csv(csv_file, sep=', ', encoding='utf-8')
    # print(rumor_df['text'].head(5))
    rumor_df['clean_text'] = rumor_df['text'].apply(remove_punctuation)
    # print(rumor_df['clean_text'].head(5))
    stopwords = stopwords_list(stopwords_file)
    rumor_df['seg_text'] = rumor_df['clean_text'].apply(
        lambda x: ' '.join([w for w in list(jieba.cut(x)) if w not in stopwords]))
    # print(rumor_df['seg_text'].head(5))
    seg_text = rumor_df['seg_text']
    x_feature = np.array(seg_text)
    x_feature_list = x_feature.tolist()
    # print(x_feature_list)
    labels = rumor_df['label']
    y_label = np.array(labels)
    y_label_list = y_label.tolist()
    # print(y_label_list)
    return x_feature_list, y_label_list
    # return seg_text, labels


def Naive_Bayes(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    text_mnb = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())
                         ])
    # count_vec = CountVectorizer(input=list)
    # tfidf = TfidfTransformer(input=list)
    # X_train_counts = count_vec.fit_transform((X_train))
    # X_train_tfidf = tfidf.fit_transform(X_train_counts)
    # print(X_train_tfidf.shape)
    # X_test_count = count_vec.fit_transform(X_test)
    # X_test_tfidf = tfidf.fit_transform(X_test_count)
    # print(X_test_tfidf.shape)
    # mnb = MultinomialNB(alpha=1)
    text_mnb.fit(X_train, y_train)
    #保存模型
    joblib.dump(text_mnb, '../models/model.pkl')
    model = joblib.load('../models/model.pkl')
    y_pred = model.predict(X_test)

    accuracy_train = model.score(X_train, y_train)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    print('训练集准确率: %.6f' % accuracy_train)
    print('测试集精确度： %.6f\n测试集召回率：%.6f\n测试集f1-score: %.6f\n 测试集AUC值：%.6f' % (precision, recall, f1, AUC))


def predict(sentence):
    stopwords = stopwords_list(stopwords_file)
    model = joblib.load('../models/model.pkl')
    seg_sentence_list = []
    seg_sentence = ' '.join([w for w in list(jieba.cut(remove_punctuation(sentence))) if w not in stopwords])
    seg_sentence_list.append(seg_sentence)
    # print(seg_sentence_list)
    predict_id = model.predict(seg_sentence_list)
    if predict_id == 1:
        print('此信息为谣言：%s' % sentence)
    else:
        print('此信息为非谣言：%s' % sentence)


if __name__ == '__main__':
    x, y = processTextFeature(weibo_data_file)
    Naive_Bayes(x, y)
    sentence1 = '研究生惊现传销团伙，已获利数千万！！！'
    sentence2 = '香港是中国的一部分，中国领土完整不容分割！！'
    predict(sentence1)
    predict(sentence2)
