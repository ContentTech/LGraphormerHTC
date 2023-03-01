import os
import xml.dom.minidom
from tqdm import tqdm
import json
import re
import tarfile
import shutil
from transformers import AutoTokenizer
from collections import defaultdict
import torch
import math

def compute_labels_relation(class_num,train_data_sample):
    # sample_num = len(train_data_sample)
    # get the co-exist label num
    label_nums = []
    labels_co_num = []
    for i in range(class_num):
        label_count = 0
        co_num = [0 for _ in range(class_num)]
        for sample_label in train_data_sample:
            if i in sample_label:
                label_count +=1
                for l in sample_label:
                    co_num[l] += 1
        label_nums.append(label_count)
        labels_co_num.append(co_num)
    # compute the ratio
    label_co_ratio = []
    for i in range(class_num):
        num = label_nums[i]
        if num > 0:
            ratio = [int(n * 10 / num) for n in labels_co_num[i]]
        else:
            ratio = [0 for n in labels_co_num[i]]
        label_co_ratio.append(ratio)

    return label_co_ratio

import string
punc = string.punctuation

def pre_process(line):
    sample = eval(line)
    tokens = sample['token']
    labels = sample['label']
    tokens_new = []
    for i in range(len(tokens)):
        token = tokens[i]
        #
        punct = str.maketrans({key: "" for key in string.punctuation})
        token = token.translate(punct)
        #
        if token.isdigit() or len(token) <= 1:
            continue
        #
        if token in punc:
            continue
        #
        if token in ['//']:
            continue

        tokens_new.append(token)

    return {'token':tokens_new,'label':labels}


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


train_file = 'rcv1_train.json'
train_data_sample = []
data = []
with open(train_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        sample = eval(line)
        train_data_sample.append(sample['label'])

        line = pre_process(line)
        data.append(line)

value_dict = torch.load('value_dict.pt')

# get coocurrence relations
class_num = len(value_dict)
label_co_ratio = compute_labels_relation(class_num, train_data_sample)
torch.save(label_co_ratio,'label_co_ratio.pt')
print('finish output the label coocurrence ')

# get the key words
tf_idf_dict = {}
tf_idf_data = TF_IDF(data)

topN_keywords_dict = {}
for key in value_dict.keys():
    top_N = get_label_key_words(data,tf_idf_data,key,top_N=5)
    topN_keywords_dict[key] = top_N

# output the result
for key in value_dict.keys():
    print(key, ':', value_dict[key],':', topN_keywords_dict[key])

torch.save(topN_keywords_dict, 'topN_keywords_dict.pt')
print('finish output the label keywords ')