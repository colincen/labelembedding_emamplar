import torch
import numpy as np
class BuildEmb:
    def __init__(self):
        pass
    @staticmethod
    def buildembedding(embedding, word2Idx, label2Idx, description, exemplar, embedding_method,
                                encoder_method, device):
        _I = np.array([1, 0, 0]).astype(np.float32)
        _O = np.array([0, 1, 0]).astype(np.float32)
        _B = np.array([0, 0, 1]).astype(np.float32)
        embeddingDim = np.shape(embedding)[1]
        Idx2label = {v: k for k, v in label2Idx.items()}
        src_labels = [Idx2label[i] for i in range(len(Idx2label))]

        tempembedding = {}
        labelembedding = []
        if embedding_method == 'description':
            slot2Id = {}
            for label in src_labels:
                if len(label) > 2 and label[1] == '-':
                    slot = label[2:]
                    descs = description[slot]
                    if slot not in slot2Id:
                        slot2Id[slot] = []
                        for token in descs:
                            slot2Id[slot].append(word2Idx[token])

            if encoder_method == 'wordembedding':
                for label in src_labels:
                    if label == 'O':
                        labelembedding.append(np.concatenate((_O, np.zeros(embeddingDim)), 0))
                    else:
                        v0 = None
                        if label[0] == 'B':
                            v0 = _B
                        elif label[0] == 'I':
                            v0 = _I
                        else:
                            v0 = _O
                        temp = []
                        for t in slot2Id[label[2:]]:
                            temp.append(embedding[t])
                        temp = sum(temp) / len(temp)
                        labelembedding.append(np.concatenate((v0, temp), 0))

        elif embedding_method == 'exemplar':
            if encoder_method == 'wordembedding':
                slot2Id = set()
                slot2embedding = {}
                # exemplar_num = len(list(exemplar.values())[0])
                for label in src_labels:
                    if len(label) > 2 and label[1] == '-':
                        slot = label[2:]
                        if slot not in slot2Id:
                            slot2Id.add(slot)

                            exemplars = exemplar[slot]
                            exemplar_num = len(exemplar[slot])
                            # print(slot)
                            Examples = []
                            for i in range(exemplar_num):
                                oneExemplar = []
                                for k in range(len(exemplars[i])):
                                    temp = []
                                    for token in exemplars[i][k]:
                                        if token not in word2Idx:
                                            if token not in tempembedding:
                                                te = np.random.randn(embeddingDim)
                                                tempembedding[token] = te
                                            temp.append(tempembedding[token])
                                        else:
                                            temp.append(embedding[word2Idx[token]])


                                        # temp.append(embedding[word2Idx.get(token, word2Idx['<PAD>'])])
                                    if (len(temp) == 0):
                                        temp = embedding[word2Idx['<PAD>']]
                                    else:
                                        temp = sum(temp) / len(temp)
                                    oneExemplar.append(temp)
                                oneExemplar = np.concatenate(oneExemplar, 0)
                                t = np.linalg.norm(oneExemplar)
                                oneExemplar = oneExemplar / t
                            Examples.append(oneExemplar)
                            slot2embedding[slot] = sum(Examples) / len(Examples)

                for label in src_labels:
                    if label == 'O':
                        labelembedding.append(np.concatenate((_O, np.zeros(3 * embeddingDim)), 0))
                    else:
                        v0 = None
                        if label[0] == 'B':
                            v0 = _B
                        elif label[0] == 'I':
                            v0 = _I
                        else:
                            v0 = _O
                        temp = slot2embedding[label[2:]]
                        labelembedding.append(np.concatenate((v0, temp), 0))

            elif encoder_method == 'wordembedding_all_sentence':
                slot2Id = set()
                slot2embedding = {}
                for label in src_labels:
                    if len(label) > 2 and label[1] == '-':
                        slot = label[2:]
                        if slot not in slot2Id:
                            slot2Id.add(slot)

                            exemplars = exemplar[slot]
                            exemplar_num = len(exemplar[slot])
                            Examples = []
                            for i in range(exemplar_num):
                                temp = []
                                oneExemplar = exemplars[i][0] + exemplars[i][1] + exemplars[i][2]
                                for token in oneExemplar:
                                    if token not in word2Idx:
                                        if token not in tempembedding:
                                            tempembedding[token] = np.random.randn(embeddingDim)
                                        temp.append(tempembedding[token])
                                    else:
                                        temp.append(embedding[word2Idx[token]])
                                    # temp.append(embedding[word2Idx.get(token, word2Idx['<PAD>'])])
                                temp = sum(temp) / len(temp)
                                t = np.linalg.norm(temp)
                                temp /= t
                                if t == 0:
                                    continue
                                Examples.append(temp)

                            if len(Examples) > 0:
                                slot2embedding[slot] = sum(Examples) / len(Examples)
                            else:
                                slot2embedding[slot] = np.random.randn(embeddingDim)

                for label in src_labels:
                    if label == 'O':
                        labelembedding.append(np.concatenate((_O, np.zeros(embeddingDim)), 0))
                    else:
                        v0 = None
                        if label[0] == 'B':
                            v0 = _B
                        elif label[0] == 'I':
                            v0 = _I
                        else:
                            v0 = _O
                        temp = slot2embedding[label[2:]]
                        labelembedding.append(np.concatenate((v0, temp), 0))

            elif encoder_method == 'wordembedding_slot_val':
                slot2Id = set()
                slot2embedding = {}
                for label in src_labels:
                    if len(label) > 2 and label[1] == '-':
                        slot = label[2:]
                        if slot not in slot2Id:
                            slot2Id.add(slot)

                            exemplars = exemplar[slot]
                            exemplar_num = len(exemplar[slot])
                            Examples = []
                            for i in range(exemplar_num):
                                temp = []
                                oneExemplar = exemplars[i][1]
                                for token in oneExemplar:
                                    if token not in word2Idx:
                                        if token not in tempembedding:
                                            tempembedding[token] = np.random.randn(embeddingDim)
                                        temp.append(tempembedding[token])
                                    else:
                                        temp.append(embedding[word2Idx[token]])

                                temp = sum(temp) / len(temp)
                                t = np.linalg.norm(temp)
                                if t == 0:
                                    continue
                                temp /= t
                                Examples.append(temp)
                            if len(Examples) > 0:
                                slot2embedding[slot] = sum(Examples) / len(Examples)
                            else:
                                slot2embedding[slot] = np.random.randn(embeddingDim)



                for label in src_labels:
                    if label == 'O':
                        labelembedding.append(np.concatenate((_O, np.zeros(embeddingDim)), 0))
                    else:
                        v0 = None
                        if label[0] == 'B':
                            v0 = _B
                        elif label[0] == 'I':
                            v0 = _I
                        else:
                            v0 = _O
                        temp = slot2embedding[label[2:]]
                        labelembedding.append(np.concatenate((v0, temp), 0))

        labelembedding = torch.tensor(labelembedding, dtype=torch.float32, device=device)
        labelembedding.requires_grad = False
        # print(labelembedding)
        return labelembedding





