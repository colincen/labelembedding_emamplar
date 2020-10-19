import torch
import numpy as np
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
    slot2embedding = {}

    if embedding_method == 'exemplar':
        if encoder_method == 'wordembedding':
            slot2Id = set()

            for label in src_labels:
                if len(label) > 2 and label[1] == '-':
                    slot = label[2:]
                    if slot not in slot2Id:
                        slot2Id.add(slot)
                        slot2embedding[slot] = []

                        exemplars = exemplar[slot]
                        exemplar_num = len(exemplar[slot])

                        # print(slot)
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

                            slot2embedding[slot].append(len(labelembedding))
                            labelembedding.append(oneExemplar)


    # labelembedding = torch.tensor(labelembedding, dtype=torch.float32, device=device)
    # labelembedding.requires_grad = False

    return labelembedding, slot2embedding