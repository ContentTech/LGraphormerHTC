from transformers import AutoTokenizer
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xml.dom.minidom
import pandas as pd
import json
from collections import defaultdict


file = '/Users/hejian.ls/Documents/fortune_content_AI/category_cls/HMCN/public_dataset/RCV1/lyrl2004_tokens_train.dat'

train_ids = []
with open(file, 'r') as f:
    for line in f.readlines():
        if line.startswith('.I'):
            train_ids.append(int(line[3:-1]))

print(len(train_ids))


train_file = 'rcv1_train.json'
dev_file = 'rcv1_dev.json'

with open(train_file, 'r') as f:
    train_lines = f.readlines()
    train_num = len(train_lines)
    print('rcv1 train num:', train_num)

with open(dev_file, 'r') as f:
    dev_lines = f.readlines()
    dev_num = len(dev_lines)
    print('rcv1 dev num:', dev_num)

print('real data num:', train_num+dev_num)


test_file = 'rcv1_test.json'

with open(test_file, 'r') as f:
    test_lines = f.readlines()
    test_num = len(test_lines)
    print('rcv1 test num:',test_num)

# index = 0
# for line in test_lines:
#     print(index)
#     index += 1
#     if line in train_lines:
#         print('same data in train')
#         break

source = []
labels = []
label_dict = {}
hiera = defaultdict(set)
with open('rcv1.taxonomy', 'r') as f:
    label_dict['Root'] = -1
    for line in f.readlines():
        line = line.strip().split('\t')
        for i in line[1:]:
            if i not in label_dict:
                label_dict[i] = len(label_dict) - 1
            hiera[label_dict[line[0]]].add(label_dict[i])
    label_dict.pop('Root')
    hiera.pop(-1)
value_dict = {i: v for v, i in label_dict.items()}
# torch.save(value_dict, 'value_dict.pt')
# torch.save(hiera, 'slot.pt')

print(label_dict)
print(value_dict)
sample_num_train = [0 for key in label_dict.keys()]
sample_num_test = [0 for key in label_dict.keys()]

for line in train_lines:
    doc = eval(line)['token']
    label = eval(line)['label']
    for i in label:
        sample_num_train[i] +=1

for line in test_lines:
    doc = eval(line)['token']
    label = eval(line)['label']
    for i in label:
        sample_num_test[i] +=1

data_dist = {'label':[], 'train_num':[],'test_num':[]}
for i in range(len(sample_num_train)):
    data_dist['label'].append(value_dict[i])
    data_dist['train_num'].append(sample_num_train[i])
    data_dist['test_num'].append(sample_num_test[i])

print(data_dist)

import pandas as pd
from  pandas import DataFrame

data_pd = DataFrame(data_dist)

data_pd.sort_values(by="train_num", ascending=False,inplace=True)
data_pd.reset_index(drop=True, inplace=True)
print(data_pd.head())
for i in range(len(data_pd)):
    print(i,'  ',data_pd.at[i,'label'],':',data_pd.at[i,'train_num'],data_pd.at[i,'test_num'])






