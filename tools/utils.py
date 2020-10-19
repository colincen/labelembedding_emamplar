import os
import os.path as op
import gzip
from typing import List

import numpy as np
import json
import random
import copy
import math
import tools.DataSet.SNIPS as SNIPS

def cal_maxlen(data):
    return max([len(x) for x in data])


def padData(data, max_len, padding_idx):
    padded = []
    for i in range(len(data)):
        temp = []
        pad = [padding_idx] * (max_len - len(data[i]))
        for token in data[i]:
            temp.append(token)
        temp += pad
        padded.append(temp)
    return padded


def prepare_data(sents):
    sentences = []
    for i in sents:
        if len(i) > 0:
            sentences.append(i)
    sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
    lengths = [len(i) for i in sentences]
    return sentences, lengths


def getNERdata(dataSetName='SNIPS',
               dataDir='/home/sh/data/JointSLU-DataSet/formal_snips',
               desc_path='../data/snips_slot_description.txt',
               cross_domain=True,
               exemplar_num=1,
               target_domain='PlayMusic'):

    if not os.path.exists(dataDir):
        raise Exception('data file not exists')

    if dataSetName == 'SNIPS':
        snips = SNIPS.snips(dataDir=dataDir, desc_path=desc_path, cross_domain=True,
                                          exemplar_num=exemplar_num,
                                          target_domain=target_domain)
        return snips.data


def ExtractLabelsFromTokens(data):
    Labels = {}
    for row in data:
        for token in row[1]:
            # print(token)
            if token not in Labels:
                Labels[token] = len(Labels)

    return Labels


def readTokenEmbeddings(embeddingsPath):
    if not op.isfile(embeddingsPath):
        print('Embedding not found : Error')
    word2Idx = {}
    embeddings = []
    neededVocab = {}

    # embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath,
    #                                                                                            encoding="utf8")
    embeddingsIn = open(embeddingsPath, 'r')
    embeddingsDimension = None
    for line in embeddingsIn:
        split = line.rstrip().split(" ")
        word = split[0]

        if embeddingsDimension == None:
            embeddingsDimension = len(split) - 1

        if (len(
                split) - 1) != embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
            print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
            continue

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["<PAD>"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension)
            embeddings.append(vector)

            word2Idx["<UNK>"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)

        vector = np.array([float(num) for num in split[1:]])

        if len(neededVocab) == 0 or word in neededVocab:
            if word not in word2Idx:
                embeddings.append(vector)
                word2Idx[word] = len(word2Idx)
    embeddings = np.array(embeddings)
    print(len(word2Idx))
    return embeddings, word2Idx

def getCharIdx():
    charset = {"<PAD>": 0, "<UNK>": 1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        charset[c] = len(charset)
    return charset

def setMapping(data, mapping):
    resData = []
    for line in data:
        temp = []
        for token in line:
            if token in mapping:
                temp.append(mapping[token])
            else:
                temp.append(mapping['<UNK>'])
        resData.append(temp)
    return resData


def data_generator(data, batch_size):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    x = []
    y = []
    bins = math.ceil(len(data) / batch_size)
    for i in range(bins):
        x.clear()
        y.clear()
        x_ids = index[i * batch_size:min(len(index), (i + 1) * batch_size)]
        for j in x_ids:
            x.append(data[j][0])
            y.append(data[j][1])
        yield x, y

def setCharMapping(data, mapping):
    # print(data)
    # print(mapping)
    sents = []
    for sent in data:
        # print(sent)
        words = []
        for word in sent:
            tokens = []
            for token in word:
                # print(token)
                tokens.append(mapping.get(token, mapping['<UNK>']))
            words.append(tokens)
        sents.append(words)
    # print(data)
    return sents

def padToken(data, paddingIdx, max_char_len):
    for i, sent in enumerate(data):
        for j,word in enumerate(sent):
            if len(word) > max_char_len:
                data[i][j] = word[:max_char_len]
            else:
                data[i][j] = word + [paddingIdx] * (max_char_len - len(word))

    max_len = max([len(i) for i in data])
    for i in range(len(data)):
        l = len(data[i]);
        while len(data[i]) < max_len:
            data[i].append([paddingIdx] * max_char_len)

    return data

