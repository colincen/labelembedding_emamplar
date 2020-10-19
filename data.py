import os
import os.path as op
import gzip
import numpy as np
import json
# /home/sh/data/JointSLU-DataSet/formal_snips
def getNERdata(dataPath = '/home/sh/data/JointSLU-DataSet/formal_snips'):
    if not os.path.exists(dataPath):
        raise Exception('data file not exists')

    if 'snips' in dataPath.lower():
        return getSnipsData(dataPath)

def getSnipsData(dataPath, intent=None):
    def getRawSentences(dataPath, wordColumn=0, labelColumn=1):
        rawText = []
        sentence = []
        label = []
        intent = []

        for line in open(dataPath, 'r'):
            row = line.strip().split()

            # read the blank space
            if len(row) == 0:
                rawText.append([sentence.copy(), label.copy(), intent.copy()])
                sentence.clear()
                label.clear()

            # read the intent
            elif len(row) == 1:
                intent = row

            # read the word and label
            elif len(row) == 2:
                sentence.append(row[wordColumn])
                label.append(row[labelColumn])

        rawText.append([sentence.copy(), label.copy(), intent.copy()])

        return rawText

    def prepareSentencesByIntent(dataPath, wordColumn=0, labelColumn=1):
        rawData = getRawSentences(dataPath, wordColumn, labelColumn)
        print('total sentence from %s is %d' % (op.basename(dataPath), len(rawData)))
        intents = set(row[2][0] for row in rawData)
        data = {intentName: {'samples': 0, 'sentences': []} for intentName in intents}

        for sent in rawData:
            data[sent[2][0]]['samples'] += 1
            data[sent[2][0]]['sentences'] .append((sent[0], sent[1]))

        return data


    train = prepareSentencesByIntent(dataPath+'/train.txt')
    dev = prepareSentencesByIntent(dataPath+'/dev.txt')
    test = prepareSentencesByIntent(dataPath+'/test.txt')
    if intent is None:
        return (train, dev, test)
    else:
        return (train[intent]['sentences'],dev[intent]['sentences'],test[intent]['sentences'])

def readEmbeddings(embeddingsPath = '/home/sh/data/komninos_english_embeddings.gz'):
    if not op.isfile(embeddingsPath):
        print('Embedding not found : Error')
    embeddings_path = '/home/sh/code/labelembedding/data/embeddings.npy'
    vocab_path = '/home/sh/code/labelembedding/data/vocab.json'
    word2Idx = {}
    embeddings = []
    neededVocab = {}

    if op.exists(embeddings_path) and op.exists(vocab_path):
        with open(vocab_path,'r') as f:
            word2Idx = json.load(f)
        f.close()
        embeddings = np.load(embeddings_path)
        return embeddings, word2Idx

    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath,
                                                                                               encoding="utf8")
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
    with open(vocab_path,'w') as f:
        json.dump(word2Idx,f)
    np.save(embeddings_path,embeddings)
    return embeddings, word2Idx

def getBIOdict(da, intent=None, labelPath = '/home/sh/code/labelembedding/data/label'):
    if intent is None:
        labelPath+='.json'
    else:
        labelPath+=intent+'.json'
    if op.exists(labelPath):
        with open(labelPath,'r') as f:
            return json.load(f)

    NER_BIO={}
    for line in da:
        BIOS = line[1]
        for label in BIOS:
            if label not in NER_BIO:
                NER_BIO[label] = len(NER_BIO)



    with open(labelPath,'w') as f:
        json.dump(NER_BIO,f)
    f.close()
    return NER_BIO

def word2num(da, word2Idx, NER_BIO):
    data = []
    label = []
    tot = 0
    knw = 0

    for line in da:
        tokens = line[0]
        tags = line[1]
        tempdata = []
        templabel = []

        for token in tokens:
            tot += 1
            word = word2Idx['<UNK>']
            if token in word2Idx:
                word = word2Idx[token]
                knw += 1
            tempdata.append(word)
        data.append(tempdata)
        for lab in tags:
            templabel.append(NER_BIO[lab])
        label.append(templabel)

    print('converge: %.3f' % (knw/tot))
    return data, label

if __name__ == '__main__':
    train,dev,test = getSnipsData('/home/sh/data/JointSLU-DataSet/formal_snips', intent = 'PlayMusic')
    w = getBIOdict(train, intent = 'PlayMusic')
    emb, word2Idx = readEmbeddings()
    # print(word2Idx['<UNK>'])
    da,lab=word2num(train,word2Idx,w)
    print(da[0:5])
    print(lab[0:5])