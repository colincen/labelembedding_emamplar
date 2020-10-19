import os.path as op
import os
import random
import copy
import json


class snips:
    def __init__(self, dataDir, desc_path, cross_domain=False, target_domain='AddToPlaylist', exemplar_num=1):
        self.data = None
        self.source = {'train': [], 'dev': [], 'test': []}
        self.target = {'train': [], 'dev': [], 'test': []}
        self.description = None
        self.exemplar = None
        self.exemplar_num = exemplar_num
        self.dataDir = dataDir
        self.train = self.prepareSentencesByIntent('train.txt')
        self.dev = self.prepareSentencesByIntent('dev.txt')
        self.test = self.prepareSentencesByIntent('test.txt')
        # SourceSlot = {'AddToPlaylist' : ['PlayMusic'],
        #               'BookRestaurant':['GetWeather','SearchScreeningEvent'],
        #               'GetWeather':['BookRestaurant','SearchScreeningEvent'],
        #               'PlayMusic':['AddToPlaylist'],
        #               'RateBook':['SearchCreativeWork', 'SearchScreeningEvent'],
        #               'SearchCreativeWork':['RateBook', 'SearchScreeningEvent'],
        #               'SearchScreeningEvent':['RateBook', 'SearchCreativeWork', 'BookRestaurant','GetWeather']}
        self.description = self.getDescription(desc_path)

        # print(self.exemplar)
        if not cross_domain:
            self.source['train'] = self.train[target_domain]
            self.source['dev'] = self.dev[target_domain]
            self.source['test'] = self.test[target_domain]
        else:
            # self.target['test'] = self.test[target_domain] + self.train[target_domain] + self.dev[target_domain]
            for k in self.train.keys():
                if k != target_domain:
                    self.source['train'] += self.train[k]
                    self.source['train'] += self.dev[k]
                    self.source['train'] += self.test[k]

            self.target['dev'] = self.dev[target_domain]


            self.target['test'] =  self.train[target_domain] + self.test[target_domain]


        # print(self.exemplar_dev)

        self.exemplar_train = self.getExemplar(self.source['train'])
        self.exemplar_dev = self.getExemplar(self.target['dev'])
        self.exemplar_test = self.getExemplar(self.target['test'])


        # Path = target_domain+'.txt'
        # if not os.path.exists(Path):
        #     self.exemplar_train = self.getExemplar(self.source['train'])
        #     self.exemplar_dev = self.getExemplar(self.target['dev'])
        #     self.exemplar_test = self.getExemplar(self.target['test'])
        #     f = open(Path, 'w')
        #     f.write(json.dumps(self.exemplar_train)+'\n')
        #     f.write(json.dumps(self.exemplar_dev) + '\n')
        #     f.write(json.dumps(self.exemplar_test))
        #     f.close()
        # else:
        #     f= open(Path,'r')
        #     for i, line in enumerate(f):
        #         line = line.strip()
        #         if i == 0:
        #             self.exemplar_train = json.loads(line)
        #         elif i == 1:
        #             self.exemplar_dev = json.loads(line)
        #         elif i == 2:
        #             self.exemplar_test = json.loads(line)


        # print(self.exemplar_train)
        # print('-'*30)
        # print(self.exemplar_dev)
        # print('-'*30)
        # print(self.exemplar_test)
        # print('-'*30)
        self.data = {'source': self.source, 'target': self.target, 'description': self.description,
                     'exemplar_train': self.exemplar_train, 'exemplar_dev': self.exemplar_dev,
                     'exemplar_test': self.exemplar_test}



    def getRawSentences(self, path):
        rawText = []
        sentence = []
        label = []
        intent = []

        for line in open(op.join(self.dataDir, path), 'r'):
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
                sentence.append(row[0])
                label.append(row[1])

        rawText.append([sentence.copy(), label.copy(), intent.copy()])

        return rawText

    def prepareSentencesByIntent(self, path):
        rawData = self.getRawSentences(path)
        print('total sentence from %s is %d' % (op.join(op.basename(self.dataDir), path), len(rawData)))
        intents = set(row[2][0] for row in rawData)
        data = {intentName: [] for intentName in intents}

        for sent in rawData:
            data[sent[2][0]].append([sent[0], sent[1]])

        return data

    def getDescription(self, path):
        slot2description = {}
        for line in open(path, 'r'):
            pos = line.find(':')
            slot = line[:pos].strip(' ')
            desc = line[pos + 1:].strip().split(' ')
            slot2description[slot] = desc
        return slot2description

    def prepareDataBySlot(self, da):
        slot2exemplar = {}
        for line in da:
            data, label = line[0], line[1]
            assert len(data) == len(label)
            i = 0
            while i < len(data):
                fg = False
                if label[i][0] == 'B':
                    fg = True
                    start = i
                    end = i + 1
                    while end < len(data) and label[end][0] == 'I':
                        end += 1
                    end = min(end, len(data))
                    slotName = label[i][2:]
                    temp = [[], [], []]
                    if slotName not in slot2exemplar:
                        slot2exemplar[slotName] = []
                    for j in range(0, start):
                        temp[0].append(data[j])
                    for j in range(start, end):
                        temp[1].append(data[j])
                    for j in range(end, len(data)):
                        temp[2].append(data[j])
                    assert len(temp) == 3
                    slot2exemplar[slotName].append(temp)
                    i = end

                if not fg:
                    i += 1
        return slot2exemplar

    def getExemplar(self, da):
        # print(len(da))
        # data = self.prepareDataBySlot('train')

        slot2exemplar = {}
        dx = copy.deepcopy(da)
        d = self.prepareDataBySlot(dx)

        # for k in d:
        #     slot2exemplar[k] = random.sample(d[k], self.exemplar_num)

        return d



if __name__ == '__main__':

    # print('ok')
    s = snips(dataDir='/home/sh/data/JointSLU-DataSet/formal_snips', desc_path='../../data/snips_slot_description.txt', cross_domain=True)
    # print(len(s.exemplar.keys()))