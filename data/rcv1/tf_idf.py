
import torch
import pandas as pd
import math

train_file = 'rcv1_train.json'
dev_file = 'rcv1_dev.json'
data = []

import string
punc = string.punctuation
def pre_process(line):
    sample = eval(line)
    tokens = sample['token']
    labels = sample['label']
    tokens_new = []
    for i in range(len(tokens)):
        token = tokens[i]
        # 去掉标点符号
        punct = str.maketrans({key: "" for key in string.punctuation})
        token = token.translate(punct)
        # 去掉数字和单个字母
        if token.isdigit() or len(token) <= 1:
            continue
        # 去掉单个标点符号
        if token in punc:
            continue
        # 去掉其他badcase
        if token in ['//']:
            continue
        tokens_new.append(token)
    return {'token':tokens_new,'label':labels}


# read data
with open(train_file, 'r') as f:
    train_lines = f.readlines()
    for line in train_lines:
        line = pre_process(line)
        data.append(line)
with open(dev_file, 'r') as f:
    dev_lines = f.readlines()
    for line in dev_lines:
        line = pre_process(line)
        data.append(line)


def TF_IDF(data):
    tf_idf_data = []
    # tf
    tf_data = []
    for i in range(len(data)):
        sample = data[i]
        tokens = sample['token']
        token_num = {}
        for item in tokens:
            if item in token_num:
                token_num[item] += 1
            else:
                token_num[item] = 1
        text_lenth = len(tokens)
        token_tf = {}
        for key in token_num.keys():
            token_tf[key] = token_num[key]/text_lenth
        tf_data.append(token_tf)
        # print(token_tf)
        # exit()
    # idf
    text_num = len(tf_data)
    token_text_num_dict = {}
    for item in tf_data:
        for token in item.keys():
            if token in token_text_num_dict:
                token_text_num_dict[token] += 1
            else:
                token_text_num_dict[token]=1
    idf_dict = {}
    for token in token_text_num_dict.keys():
        idf_dict[token] = math.log(text_num/(token_text_num_dict[token]+1))
    # tf_idf
    for i in range(len(tf_data)):
        sample = tf_data[i]
        tf_idf_sample_dict = {}
        for key in sample.keys():
            tf_idf_sample_dict[key] = sample[key]*idf_dict[key]
        tf_idf_data.append(tf_idf_sample_dict)

    return tf_idf_data

def get_label_key_words(data, tf_idf_data, label, top_N=10):
    label_keywords_weight_dict = {}
    for i in range(len(data)):
        sample = data[i]
        if label in sample['label']:
            sample_tf_idf = tf_idf_data[i]
            for token in sample_tf_idf.keys():
                if token in label_keywords_weight_dict:
                    label_keywords_weight_dict[token] = label_keywords_weight_dict[token]+sample_tf_idf[token]
                else:
                    label_keywords_weight_dict[token] = sample_tf_idf[token]

    # return top N
    keywords_sorted_dict = sorted(label_keywords_weight_dict.items(), key=lambda x:x[1],reverse=True)
    keywords_topN =keywords_sorted_dict[0:top_N]

    return keywords_topN

value_dict = torch.load('value_dict.pt')
print(value_dict)
tf_idf_dict = {}
tf_idf_data = TF_IDF(data)

topN_keywords_dict = {}
for key in value_dict.keys():
    top_N = get_label_key_words(data,tf_idf_data,key,top_N=10)
    topN_keywords_dict[key] = top_N


# output the result
for key in value_dict.keys():
    print(key, ':', value_dict[key],':', topN_keywords_dict[key])

torch.save(topN_keywords_dict, 'topN_keywords_dict.pt')
print('finish output the label keywords ')


