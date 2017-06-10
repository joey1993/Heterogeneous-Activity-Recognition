import numpy as np
import re
import json
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
import gensim


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_datasets(train_test_data):

    lines = np.array(open(train_test_data, "r").readlines())
    data_train = list()
    data_train_target = list()
    target_set = set()
    for item in lines:
        line = json.loads(item)
        data_train_target.append(line['label'])
        target_set.add(line['label'])
        g_all = list()
        a_all = list()
        for record in line['gdata']:
            g_data = list()
            for value in record['data']:
                g_data.append(float(value))
            g_all.append(g_data)
        for record in line['adata']:
            a_data = list()
            for value in record['data']:
                a_data.append(float(value))
            a_all.append(a_data)
        g_a_data = np.concatenate([g_all,a_all],1)
        g_a_input = np.reshape(np.transpose(g_a_data),[-1])
        data_train.append(g_a_input)
    datasets = dict()
    datasets['data'] = np.array(data_train)
    datasets['target'] = np.array(data_train_target)
    datasets['target_names'] = list(target_set)
    print "labels: ", datasets['target_names']
    return datasets


def load_data_labels(datasets):

    x = np.array(datasets['data'])

    # Generate labels
    labels = []
    for i in range(len(x)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target_names'].index(datasets['target'][i])] = 1
        labels.append(label)
    y = np.array(labels)
    return [x, y]
