import torch
import torch.nn as nn
import numpy as np
from tools.utils import prepare_data, setMapping, padData, cal_maxlen, setCharMapping, padToken
from models.buildlabelembedding import BuildEmb
from models.charembedding import charEmbedding
import torch.nn.utils.rnn as rnn_utils
import models.crf as crf
import copy
import os


class Bilstm_LabelEmbedding(nn.Module):
    def __init__(self, config, embedding, word2Idx, label2Idx, char2Idx, description, exemplar):
        super(Bilstm_LabelEmbedding, self).__init__()


        self.embed_size = config.embed_size
        self.max_char_len = config.max_char_len
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.Bidirectional = config.bidirectional
        self.lstm_dropout = config.lstm_dropout
        self.dropout_rate = config.dropout
        self.use_charEmbedding = config.use_charEmbedding
        self.use_crf = config.crf
        self.device = config.device
        self.word2Idx = word2Idx
        self.char2Idx = char2Idx
        self.label2Idx = label2Idx
        self.embedding = embedding
        self.description = description
        self.exemplar = exemplar
        self.label2Idx = label2Idx
        self.config = config




        self.TokenEmbedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding.astype(np.float32)),
                                                           padding_idx=word2Idx['<PAD>'])


        self.charemb = None
        if self.use_charEmbedding:

            self.charemb = charEmbedding(eval(config.conv_filter_sizes), eval(config.conv_filter_nums), char2Idx, config.char_emb_size, self.device)



        self.LabelEmbedding = self.init_LabelEmbedding(sz=(3,1206))


        LSTMDim = self.embed_size
        if self.use_charEmbedding:
            LSTMDim += sum(self.charemb.conv_filter_nums)
        self.Lstm = nn.LSTM(input_size=LSTMDim,
                            dropout=self.lstm_dropout,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=True,
                            batch_first=True,
                            bidirectional=self.Bidirectional)
        # self.h_projection = nn.Linear(2 * self.hidden_size if self.Bidirectional else self.hidden_size,
        #                               self.LabelEmbedding.size(1),
        #                               bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)





        self.crf = None
        if self.use_crf:
            self.crf = crf.CRF(labelEmbedding=self.LabelEmbedding, num_tags=self.LabelEmbedding.size(0),
                               batch_first=True)


    def init_LabelEmbedding(self, sz):
        return torch.randn(sz, device=self.device)


    @staticmethod
    def BuildLabelEmbedding(embedding, word2Idx, label2Idx, description, exemplar, embedding_method,
                            encoder_method, device):
        return BuildEmb.buildembedding(embedding, word2Idx, label2Idx, description, exemplar, embedding_method,
                              encoder_method, device)




    def forward(self, x, y, Type):
        x, x_lengths = prepare_data(x)
        y, y_lengths = prepare_data(y)
        if self.use_charEmbedding:
            char_pad_x = copy.deepcopy(x)
            char_pad_x = setCharMapping(char_pad_x, self.charemb.char2Idx)
            char_pad_x = padToken(char_pad_x, self.charemb.char2Idx['<PAD>'], self.max_char_len)


        # print(char_pad_x)
            char_pad_x = torch.tensor(char_pad_x, device=self.device)
        # print(char_pad_x.size())
            char_pad_x = self.charemb(char_pad_x)
            char_pad_x = self.dropout(char_pad_x)
        # print(len(char_pad_x))
        # print(char_pad_x.size())






        _x = copy.deepcopy(x)
        _y = copy.deepcopy(y)
        x = setMapping(x, self.word2Idx)
        y = setMapping(y, self.label2Idx)
        x = padData(x, cal_maxlen(x), self.word2Idx['<PAD>'])
        y = padData(y, cal_maxlen(y), self.label2Idx['O'])
        x = torch.tensor(x, device=self.device)
        y = torch.tensor(y, device=self.device)

        mask = (torch.zeros(x.size(), device=self.device).type_as(x) == x)

        mask = (mask == 0)
        mask = mask.byte()

        x = self.TokenEmbedding(x)
        if self.use_charEmbedding:
            x = torch.cat((x,char_pad_x), -1)
        # print(x.size())
        packed = rnn_utils.pack_padded_sequence(x, x_lengths, batch_first=True)
        x, _ = self.Lstm(packed)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        fw, bw = torch.split(x, [self.hidden_size, self.hidden_size],dim=-1)

        # x = self.h_projection(x)
        # batch_size x max_seq_len x (3 + |A_src|)

        # print(fw.size())
        # print(bw.size())
        # print(self.LabelEmbedding.size())

        y1 = torch.matmul(fw, self.LabelEmbedding.transpose(0, 1))
        y2 = torch.matmul(bw, self.LabelEmbedding.transpose(0, 1))
        y_hat = y1 + y2

       #  print(self.use_crf)
        if Type == 'train':
            if not self.use_crf:
                batch_size, max_len = y_hat.size(0), y_hat.size(1)
                feats = y_hat.view(batch_size * max_len, -1)
                tags = y.view(-1)
                loss_func = nn.CrossEntropyLoss(size_average=True)
                loss = loss_func(feats, tags)
                return loss
            else:

                loss = -self.crf(y_hat, y, mask, 'mean')
                return loss
        elif Type == 'test':
            y_pad = None
            if not self.use_crf:
                y_pad = y_hat.argmax(-1).detach().tolist()
            else:
                y_pad = self.crf.decode(y_hat, mask)
            pred = []
            id2label = {v: k for k, v in self.label2Idx.items()}
            for i in range(len(y_pad)):
                for j in range(len(y_pad[i])):
                    y_pad[i][j] = id2label[y_pad[i][j]]
                pred.append(y_pad[i][:x_lengths[i]])
            return _x, _y, pred

    @staticmethod
    def load(model_path, device='cpu'):
        model_params = torch.load(model_path, map_location=lambda storage, loc: storage)
        if not os.path.exists(os.path.join(os.path.dirname(model_path), 'params')):
            raise Exception('params data error')

        params_path = os.path.join(os.path.dirname(model_path), 'params')
        params = torch.load(params_path, map_location=lambda storage, loc: storage)
        config = params['config']
        word2Idx = params['word2Idx']
        embedding = params['embedding']
        description = params['description']
        exemplar = params['exemplar']
        label2Idx = params['label2Idx']
        char2Idx= params['char2Idx']
        config.device = device
        model = Bilstm_LabelEmbedding(config=config, word2Idx=word2Idx, embedding=embedding,
                                      description=description, exemplar=exemplar,
                                      label2Idx=label2Idx, char2Idx=char2Idx)




        LabelEmb  = model_params['state_dict']['crf.labelembedding.weight']
        if config.crf:
            model.crf.labelembedding = model.crf.buildCRFLabelEmbedding(LabelEmb)



        model.load_state_dict(model_params['state_dict'])
        return model

    def save(self, path):
        print('save model parameters to [%s]' % path)
        if not os.path.exists(os.path.join(os.path.dirname(path), 'params')):
            params_path = os.path.join(os.path.dirname(path), 'params')
            params = {
                'config': self.config,
                'embedding': self.embedding,
                'description': self.description,
                'exemplar': self.exemplar,
                'word2Idx': self.word2Idx,
                'label2Idx': self.label2Idx,
                'char2Idx':self.char2Idx
            }
            torch.save(params, params_path)
        model_params = {

            'state_dict': self.state_dict()
        }
        torch.save(model_params, path)

