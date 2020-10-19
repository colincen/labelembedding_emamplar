import sys
from tools.utils import *
import tools.DataSet.SNIPS as SNIPS

class Check_utils:
    @staticmethod
    def check_cal_max_len():
        data = [[1, 2, 3, 4, 5], [1, 2], [4, 5, 6, 7]]
        max_len = cal_maxlen(data)
        if max_len == 5:
            print('utils.cal_max_len check passed!', file=sys.stderr)

    @staticmethod
    def check_readLabelEmbddding():
        data = getNERdata()
        Labels = ExtractLabelsFromTokens(data['source']['train'])
        # e, l = readLabelEmbedding(description=data['description'],embeddingsPath='/home/sh/data/komninos_english_embeddings.gz',Labels=Labels)
        # print(l)

    @staticmethod
    def check_data_gen():
        data = getNERdata()
        for i in data_generator(data['source']['train'], 2):
            print(i)

    @staticmethod
    def check_setmapping():
        data = getNERdata()['source']['train']
        Labels = ExtractLabelsFromTokens(data)
        e, l = readLabelEmbedding(embeddingsPath='/home/sh/data/komninos_english_embeddings.gz', Labels=Labels)
        for i in data_generator(data, 2):
            print(i)
            da = i[1]
            print(da)
            print(l)
            res = setMapping(da, l)
            print(res)
            print('----------')

class Check_Snips:
    def __init__(self):
        self.Snips = SNIPS.snips(dataDir= '/home/sh/data/JointSLU-DataSet/formal_snips', desc_path= '../data/snips_slot_description.txt', cross_domain=True)

    def check_getDescription(self):
        print(self.Snips.description)

    def check_getExempalr(self):
        for k, v in self.Snips.exemplar.items():
            print(k)
            print(len(v))
            print('--------------------------')

    def check_data(self):
        print(len(self.Snips.data['exemplar']))

c = Check_Snips()
print(c.Snips.data['source']['train'])