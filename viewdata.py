import matplotlib.pyplot as plt
import json
intent = ['AddToPlaylist','BookRestaurant','GetWeather','PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']
filepath = 'data/additive/'
x =[0, 1, 2, 3, 4]
for i in intent:
    j = filepath+i
    f = open(j+'/result.txt', 'r')
    temp = []
    for k, line in enumerate(f):
        d = json.loads(line)
        temp.append(d['total']['f1'])
    plt.bar(x, temp, align = 'center')
    plt.title(i)
    plt.xlabel('times')
    plt.ylabel('f1')
    plt.show()

















#
# from model.bilstm import Bilstm_LabelEmbedding
# from tools.DataSet.SNIPS import snips
# from tools.utils import *
# import torch
# import copy
# from tools.conll import evaluate
# target_domain = 'GetWeather'
# model_path = '/home/sh/code/labelembedding/data/GetWeather/model'
# device = 'cpu'
#
# slots = ['album', 'artist', 'best_rating', 'city', 'condition_description',
#          'condition_temperature', 'country', 'cuisine', 'current_location', 'entity_name',
#          'facility', 'genre', 'geographic_poi', 'location_name', 'movie_name', 'movie_type',
#          'music_item', 'object_location_type', 'object_name', 'object_part_of_series_type', 'object_select',
#          'object_type', 'party_size_description', 'party_size_number', 'playlist', 'playlist_owner', 'poi',
#          'rating_unit', 'rating_value', 'restaurant_name', 'restaurant_type', 'served_dish', 'service',
#          'sort', 'spatial_relation', 'state', 'timeRange', 'track', 'year']
#
# model = Bilstm_LabelEmbedding.load(model_path, device)
# config = model.config
# dataDict = getNERdata(dataSetName=config.dataset,
#                           dataDir=config.data_dir,
#                           desc_path=config.description_path,
#                           cross_domain=config.cross_domain,
#                           exemplar_num=config.exemplar_num,
#                           target_domain=config.target_domain)
# # temp = dataDict['description']
# # label2Idx = {}
# # for k,v in temp.items():
# #     label2Idx[k] = len(label2Idx)
# tgt_label2Idx = ExtractLabelsFromTokens(dataDict['target']['test'])
# # tgt_label2Idx = label2Idx
# # tgt_label2Idx = ExtractLabelsFromTokens(dataDict['source']['train'])
# print(tgt_label2Idx)
# model.label2Idx = tgt_label2Idx
# model.LabelEmbedding = Bilstm_LabelEmbedding.BuildLabelEmbedding(model.embedding, model.word2Idx, tgt_label2Idx,
#                                                                  model.description, model.exemplar, 'description',
#                                                                  'wordembedding', device)
# if config.crf:
#     model.crf.labelembedding = model.crf.buildCRFLabelEmbedding(model.LabelEmbedding)
#     model.crf.num_tags = model.LabelEmbedding.size(0)
# model.to(device)
#
#
# # with open('rawtest.txt', 'w') as f:
# #     for i in dataDict['target']['test']:
# #         f.write(str(i['tokens'])+'\n')
# #         f.write(str(i['NER_BIO'])+'\n')
# #         f.write(str(i['slot'])+'\n')
# #         f.write('\n')
# # f.close()
#
# # temple = [[['put', 'lindsey', 'cardinale', 'into', 'my', 'hillary', 'clinton', 's', 'women', 's', 'history', 'month', 'playlist'],
# #         ['O', 'B-entity_name', 'I-entity_name', 'O', 'B-playlist', 'I-playlist', 'I-playlist', 'I-playlist', 'I-playlist', 'O']]]
#
# data = dataDict['target']['test']
#
# _gold = []
# _pred = []
# model.eval()
# with torch.no_grad():
#     for pa in data_generator(data, 1):
#         if len(pa) <= 0:
#             continue
#         if len(pa[0]) <= 0:
#             continue
#         x = pa[0]
#         if len(x[0]) <= 0:
#             continue
#         y = pa[1]
#
#         p = model.Eval(x)
#         # print(y[0])
#         # print(p[0])
#         _gold += y[0]
#         _pred += p[0]
#
#
# _gold = [k[0] for k in _gold]
# _pred = [k[0] for k in _pred]
#
# def addSlot(a):
#     _a = []
#     for i in a:
#         if i == 'O':
#             _a.append(i)
#         else:
#             _a.append(i[0]+'-a')
#     return _a
# _gold = addSlot(_gold)
# _pred = addSlot(_pred)
#
# evaluate(_gold, _pred)
# _cnt1 = 0
# for a in _gold:
#     if a == 'O':
#         _cnt1+=1
# _cnt2 = 0
# for a in _pred:
#     if a == 'O':
#         _cnt2+=1
# print(_cnt1)
# print(_cnt2)