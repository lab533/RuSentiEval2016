import numpy as np
import os
from keras.preprocessing import sequence
import itertools


def nums_to_one_hot(nums, dim):
    one_hot = np.zeros((len(nums), dim), dtype=int)
    for i, num in enumerate(nums):
        one_hot[i, num] = 1
    return one_hot

  
def get_patterns(sequences, patterns, stride=5):
    starts = np.arange(0, sequences.shape[1], stride)    
    new_sentences = [[] for p in patterns]
    pattern_idxs = [[start+offset for start, offset in itertools.product(pattern, starts)]
                        for pattern in patterns] 
    for seq in sequences:        
        [new_sentences[idx].append(seq[np.sort(indexies)]) for idx, indexies in enumerate(pattern_idxs)]    
    for idx in range(len(new_sentences)):
        new_sentences[idx] = np.array(new_sentences[idx])
    return new_sentences


def read_from_file(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            sentences.append((line.lower()).split())
            line = f.readline()
    return sentences


def load_interim_data(prefix):
    ''' 
    Return data, labels, token2id
    data: (sentences, ancestors, siblings)
    labels: list of tuples (dataset name, 1 if train, 0 else, sentiment class)
    token2id: mapping token to id        
    '''
    path = os.path.join(prefix, 'sentences.txt')
    sib_path = os.path.join(prefix, 'siblings.txt')
    anc_path = os.path.join(prefix, 'ancestors.txt')        
    label_path = os.path.join(prefix, 'labels.txt')
    vocab_path = os.path.join(prefix, 'token2id.txt' )    
    data = (read_from_file(path), read_from_file(anc_path), read_from_file(sib_path))
    
    label_lines = [line.split('\t') 
                   for line in open(label_path, 'r').read().strip('\n').split('\n')]
    labels = [(line[0], int(line[1]), int(line[2])) for line in label_lines]    
    
    vocab_lines = [line.split('\t') 
                   for line in open(vocab_path, 'r').read().strip('\n').split('\n')]
    token2id = dict([(line[0], int(line[1])) for line in vocab_lines])
    return data,  labels, token2id


def set_padding(data, lens=[50]):         
    padded = [sequence.pad_sequences(feature, maxlen=maxlen) 
              for feature, maxlen in zip(data, lens)]
    return padded


def split_data(X, y, case):
    train = np.array([i for i in range(len(case)) if case[i] == 1], dtype=int)
    test = np.array([i for i in range(len(case)) if case[i] == 2], dtype=int)
    X_train = [feature[train] for feature in X]
    X_test = [feature[test] for feature in X]
    y_train = y[train]
    y_test = y[test]
    return X_train, X_test, y_train, y_test


def get_sample_case(label, train_data, test_data):
    if (label[1] == 1) and (label[0] in train_data):
        return 1
    elif (label[1] == 0) and (label[0] in test_data):
        return 2
    return 0


sentences_to_ids = lambda sentences, vocab: [[vocab[token] for token in sent] for sent in sentences]  