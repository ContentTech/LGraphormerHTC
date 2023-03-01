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

np.random.seed(7)


if __name__ == '__main__':
    source = []
    labels = []
    label_dict = {}
    hiera = defaultdict(set)
    with open('data/wos.taxnomy', 'r') as f:
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
    torch.save(value_dict, 'value_dict.pt')
    torch.save(hiera, 'slot.pt')

    print(label_dict)

    file_test = 'data/wos_test.json'
    file_train = 'data/wos_train.json'
    file_val = 'data/wos_val.json'

    data_set = []
    with open(file_test, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            data = json.loads(line)
            # print(data)
            data_set.append(data)
            sample_label = data['doc_label']
            # print(sample_label)
            labels.append(sample_label)
    # print(labels)

    # train_data = []
    with open(file_train, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            data = json.loads(line)
            # print(data)
            data_set.append(data)
            sample_label = data['doc_label']
            # print(sample_label)
            labels.append(sample_label)

    # val_data = []
    with open(file_val, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            data = json.loads(line)
            # print(data)
            data_set.append(data)
            sample_label = data['doc_label']
            # print(sample_label)
            labels.append(sample_label)


    #      split into train val test
    id = [i for i in range(46985)]
    np_data = np.array(data_set)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train_data = list(train)
    print('train sample num:',len(train_data))
    val_data = list(val)
    print('val sample num:', len(val_data))
    test_data = list(test)
    print('test sample num:', len(test_data))

    with open('WebOfScience_train.json', 'w') as f:
        for item in train_data:
            line = json.dumps({'token': item['doc_token'], 'label': [label_dict[i] for i in item['doc_label']]})
            f.write(line + '\n')
    print('finish writting the train file')

    with open('WebOfScience_dev.json', 'w') as f:
        for item in val_data:
            line = json.dumps({'token': item['doc_token'], 'label': [label_dict[i] for i in item['doc_label']]})
            f.write(line + '\n')
    print('finish writting the dev file')

    with open('WebOfScience_test.json', 'w') as f:
        for item in test_data:
            line = json.dumps({'token': item['doc_token'], 'label': [label_dict[i] for i in item['doc_label']]})
            f.write(line + '\n')
    print('finish writting the test file')

    def compute_labels_relation(class_num, train_data_sample):
        # sample_num = len(train_data_sample)
        # get the co-exist label num
        label_nums = []
        labels_co_num = []
        for i in range(class_num):
            label_count = 0
            co_num = [0 for _ in range(class_num)]
            for sample_label in train_data_sample:
                if i in sample_label:
                    label_count += 1
                    for l in sample_label:
                        co_num[l] += 1
            label_nums.append(label_count)
            labels_co_num.append(co_num)
        print(label_nums)
        # compute the ratio
        sample_num = max(label_nums)
        label_ratio = []
        label_co_ratio = []
        for i in range(class_num):
            num = label_nums[i]
            label_ratio.append(int(num * 100 / sample_num))
            if num>0:
                ratio = [int(n * 100 / num) for n in labels_co_num[i]]
            else:
                ratio = [0 for n in labels_co_num[i]]
            label_co_ratio.append(ratio)

        return label_ratio, label_co_ratio


    train_file = 'WebOfScience_train.json'
    train_data_sample = []
    with open(train_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = eval(line)
            # print(sample['label'])
            train_data_sample.append(sample['label'])
    class_num = len(label_dict)
    label_ratio, label_co_ratio = compute_labels_relation(class_num, train_data_sample)
    torch.save(label_ratio, 'label_ratio.pt')
    torch.save(label_co_ratio, 'label_co_ratio.pt')
    print(label_ratio)
    for item in label_co_ratio:
        print(item)

    # analyse the distribution of the ratio

    ratio_distribution = [0 for _ in range(101)]

    for row in label_co_ratio:
        for i in row:
            ratio_distribution[i] += 1
    print(ratio_distribution)
    print(len(label_dict))

    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    plt.bar(range(len(ratio_distribution)), ratio_distribution)
    plt.title('WOS', fontsize=20)
    plt.show()