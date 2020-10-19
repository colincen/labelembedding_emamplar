import argparse
import math

from models.bilstm_additive import Bilstm_LabelEmbedding
from tools.utils import getNERdata, readTokenEmbeddings, data_generator,  ExtractLabelsFromTokens, getCharIdx
from tools.Log import Logger
import tools.conlleval as conlleval
import tools.conll as conll
import torch
import json
import torch.nn as nn
import sys
import time
import numpy as np
import os


def model_config():
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument('--dataset', type=str, default='SNIPS')
    data_arg.add_argument('--data_dir', type=str, default='/home/sh/data/JointSLU-DataSet/formal_snips')
    data_arg.add_argument('--description_path', type=str, default='data/snips_slot_description.txt')
    data_arg.add_argument("--save_dir", type=str, default='/home/sh/code/labelembedding/data/additive/')
    data_arg.add_argument("--embed_file", type=str, default='/home/sh/data/komninos_english_embeddings.gz')
    data_arg.add_argument("--vocab_path", type=str, default='/home/sh/code/labelembedding/data/vocab.json')

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=903)
    net_arg.add_argument("--bidirectional", type=bool, default=True)
    net_arg.add_argument("--lstm_dropout", type=float, default=0.5)
    net_arg.add_argument("--dropout", type=float, default=0.5)
    net_arg.add_argument("--vocab_size", type=int, default=30004)
    net_arg.add_argument("--min_len", type=int, default=1)
    net_arg.add_argument("--max_len", type=int, default=500)
    net_arg.add_argument("--max_char_len", type=int, default=16)
    net_arg.add_argument("--char_emb_size", type=int, default=8)
    net_arg.add_argument("--num_layers", type=int, default=1)
    net_arg.add_argument("--num_labels", type=int, default=3)
    net_arg.add_argument("--use_charEmbedding", type=int, default=1)
    net_arg.add_argument("--conv_filter_sizes", type=str, default='(3, 4, 5)')
    net_arg.add_argument("--conv_filter_nums", type=str, default='(30, 40, 50)')
    net_arg.add_argument("--crf", type=int, default=1)
    net_arg.add_argument("--dim3foratt", type=int, default=1024)
    net_arg.add_argument("--attn", type=str, default='dot',
                         choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--embedding_method", type=str, default='exemplar',
                         choices=['description', 'atom', 'exemplar'])
    net_arg.add_argument("--encoder_method", type=str, default='wordembedding',
                         choices=['wordembedding', 'bilstm', 'cnn', 'wordembedding_slot_val', 'wordembedding_all_sentence'])
    # Training
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--cross_domain", type=bool, default=True)
    train_arg.add_argument("--exemplar_num", type=int, default=1)
    train_arg.add_argument("--target_domain", type=str, default='AddToPlaylist')
    train_arg.add_argument("--epoch", type=int, default=10)
    train_arg.add_argument("--log_every", type=int, default=50)
    train_arg.add_argument("--log_valid", type=int, default=200)
    train_arg.add_argument("--patience", type=int, default=5)
    train_arg.add_argument("--max_num_trial", type=int, default=5)
    train_arg.add_argument("--lr_decay", type=float, default=0.5)
    train_arg.add_argument("--learning_rate", type=float, default=0.0001)
    train_arg.add_argument("--run_type", type=str, default='train')
    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--device", type=str, default='cuda:1')
    misc_arg.add_argument("--batch_size", type=int, default=32)

    config = parser.parse_args()

    return config


def evaluate(model, data, batch_size, log):
    was_training = model.training
    model.eval()
    pred = []
    gold = []
    with torch.no_grad():
        for pa in data_generator(data, batch_size):
            x = pa[0]
            y = pa[1]
            _x, _y, p = model(x, y, 'test')
            gold += _y
            pred += p

    if was_training:
        model.train()

    _gold = []
    _pred = []
    _gold_ = []
    _pred_ = []
    for i in gold:
        for j in i:
            _gold.append(j)
            if j == 'O':
                _gold_.append(j[0])
            else:
                _gold_.append(j[0]+'-a')
    for i in pred:
        for j in i:
            _pred.append(j)
            if j == 'O':
                _pred_.append(j[0])
            else:
                _pred_.append(j[0] + '-a')

    return conlleval.evaluate(_gold, _pred, log, verbose=True)


def train(config):
    dataDict = getNERdata(dataSetName=config.dataset,
                          dataDir=config.data_dir,
                          desc_path=config.description_path,
                          cross_domain=config.cross_domain,
                          exemplar_num=config.exemplar_num,
                          target_domain=config.target_domain)


    emb, word2Idx = readTokenEmbeddings(config.embed_file)
    char2Idx = getCharIdx()
    label2Idx = ExtractLabelsFromTokens(dataDict['source']['train'])
    # print(label2Idx)
    # print('---------------------------------')
    label2IdxForDev = ExtractLabelsFromTokens(dataDict['target']['dev'])
    # print(label2IdxForDev)
    # print('---------------------------------')
    label2IdxForTest = ExtractLabelsFromTokens(dataDict['target']['test'])
    # print(label2IdxForTest)
    # print('---------------------------------')
    #
    # dataDict['exemplar_dev']['music_item'] = dataDict['exemplar_train']['music_item']
    # dataDict['exemplar_dev']['artist'] = dataDict['exemplar_train']['artist']
    # dataDict['exemplar_dev']['playlist'] = dataDict['exemplar_train']['playlist']
    DevLabelEmbedding = Bilstm_LabelEmbedding.BuildLabelEmbedding(emb, word2Idx, label2IdxForDev,
                                                                     dataDict['description'], dataDict['exemplar_dev'],
                                                                     config.embedding_method,
                                                                     config.encoder_method, config.device)
    TestLabelEmbedding = Bilstm_LabelEmbedding.BuildLabelEmbedding(emb, word2Idx, label2IdxForTest,
                                                                  dataDict['description'], dataDict['exemplar_test'],
                                                                   config.embedding_method,
                                                                   config.encoder_method, config.device)

    max_batch_size = math.ceil(len(dataDict['source']['train']) / config.batch_size)


    model = Bilstm_LabelEmbedding(config, emb, word2Idx, label2Idx, char2Idx, dataDict['description'], dataDict['exemplar_train'])
    model.train()
    model = model.to(config.device)
    hist_valid_scores = []
    patience = num_trial = 0
    train_iter = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_time = time.time()

    config.save_dir = config.save_dir + config.target_domain+'/'

    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    if os.path.exists(os.path.join(config.save_dir, 'params')):
        os.remove(os.path.join(config.save_dir, 'params'))

    log = Logger(os.path.join(config.save_dir, '_src.txt'), level='info')

    for epoch in range(config.epoch):
        for da in data_generator(dataDict['source']['train'], config.batch_size):
            train_iter += 1
            x = da[0]
            y = da[1]
            loss = model(x, y, 'train')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_iter % config.log_every == 0:
                # print(
                #     'epoch %d, iter %d, loss %.2f, time elapsed %.2f sec' %
                #     (epoch, train_iter, loss, time.time() - train_time),
                #     file=sys.stderr)
                log.logger.info(
                    'epoch %d, iter %d, loss %.2f, time elapsed %.2f sec' %
                    (epoch, train_iter, loss, time.time() - train_time))

            train_time = time.time()

            if train_iter % config.log_valid == 0:
                trainLabelEmbedding = model.LabelEmbedding
                trainLabel2Idx = model.label2Idx

                model.label2Idx = label2IdxForDev
                model.LabelEmbedding = DevLabelEmbedding
                if config.crf:
                    model.crf.labelembedding = model.crf.buildCRFLabelEmbedding(model.LabelEmbedding)
                    model.crf.num_tags = model.LabelEmbedding.size(0)
                (valid_metric_pre, valid_metric_rec, valid_metric_f1), d = evaluate(model, dataDict['target']['dev'], config.batch_size, log)

                model.label2Idx = label2IdxForTest
                model.LabelEmbedding = TestLabelEmbedding
                if config.crf:
                    model.crf.labelembedding = model.crf.buildCRFLabelEmbedding(model.LabelEmbedding)
                    model.crf.num_tags = model.LabelEmbedding.size(0)
                (test_metric_pre, test_metric_rec, test_metric_f1), d = evaluate(model, dataDict['target']['test'], config.batch_size, log)

                model.label2Idx = label2Idx
                model.LabelEmbedding = trainLabelEmbedding
                if config.crf:
                    model.crf.labelembedding = model.crf.buildCRFLabelEmbedding(model.LabelEmbedding)
                    model.crf.num_tags = model.LabelEmbedding.size(0)

                # print("val_pre : %.4f, val_rec : %.4f, val_f1 : %.4f" % (valid_metric_pre, valid_metric_rec, valid_metric_f1), file=sys.stderr)
                # print("test_pre : %.4f, test_rec : %.4f, test_f1 : %.4f" % (test_metric_pre, test_metric_rec, test_metric_f1), file=sys.stderr)
                log.logger.info("val_pre : %.4f, val_rec : %.4f, val_f1 : %.4f" % (
                valid_metric_pre, valid_metric_rec, valid_metric_f1))
                log.logger.info("test_pre : %.4f, test_rec : %.4f, test_f1 : %.4f" % (
                test_metric_pre, test_metric_rec, test_metric_f1))
                is_better = len(hist_valid_scores) == 0 or valid_metric_f1 > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric_f1)
                if is_better:
                    patience = 0
                    # print('save currently the best model to [%s]' % (config.save_dir + 'model'), file=sys.stderr)
                    log.logger.info('save currently the best model to [%s]' % (config.save_dir + 'model'))
                    model.save(config.save_dir + 'model')

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), config.save_dir + 'optim')
                elif patience < config.patience:
                    patience += 1
                    log.logger.info('hit patience %d' % patience)
                    # print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(config.patience):
                        num_trial += 1
                        log.logger.info('hit #%d trial' % num_trial)
                        # print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == config.max_num_trial:
                            log.logger.info('early stop!')
                            # print('early stop!', file=sys.stderr)
                            exit(0)


                        lr = optimizer.param_groups[0]['lr'] * config.lr_decay
                        log.logger.info('load previously best model and decay learning rate to %f' % lr)
                        # print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(config.save_dir +'model', map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(config.device)

                        log.logger.info('restore parameters of the optimizers')
                        # print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(config.save_dir + 'optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0


def cross_domain(model_path, device='cuda:1'):
    model = Bilstm_LabelEmbedding.load(model_path, device)
    config = model.config
    log = Logger(os.path.join(config.save_dir, '_tgt.txt'), level='info')
    dataDict = getNERdata(dataSetName=config.dataset,
                          dataDir=config.data_dir,
                          desc_path=config.description_path,
                          cross_domain=config.cross_domain,
                          exemplar_num=config.exemplar_num,
                          target_domain=config.target_domain)
    tgt_label2Idx = ExtractLabelsFromTokens(dataDict['target']['test'])
    # labs = list(dataDict['description'].keys())
    # tgt_label2Idx = {'O' : 0}
    # for k in labs:
    #     tgt_label2Idx['B-' + k] = len(tgt_label2Idx)
    #     tgt_label2Idx['I-' + k] = len(tgt_label2Idx)
    # print(tgt_label2Idx)
    model.label2Idx = tgt_label2Idx
    model.LabelEmbedding = Bilstm_LabelEmbedding.BuildLabelEmbedding(model.embedding,model.word2Idx,tgt_label2Idx,
                                                                     model.description,dataDict['exemplar_test'],
                                                                     config.embedding_method,
                                                                     config.encoder_method,device)
    if config.crf:
        model.crf.labelembedding = model.crf.buildCRFLabelEmbedding(model.LabelEmbedding)
        model.crf.num_tags = model.LabelEmbedding.size(0)
    model.to(device)

    (test_metric_pre, test_metric_rec, test_metric_f1), d = evaluate(model, dataDict['target']['test'], config.batch_size, log)
    f = open(os.path.join(config.save_dir, 'result.txt'), 'a+')
    js = json.dumps(d)
    f.write(js+'\n')
    f.close()

    # print("test_pre : %.4f, test_rec : %.4f, test_f1 : %.4f" % (test_metric_pre, test_metric_rec, test_metric_f1),
    #       file=sys.stderr)
    log.logger.info("test_pre : %.4f, test_rec : %.4f, test_f1 : %.4f" % (test_metric_pre, test_metric_rec, test_metric_f1))



if __name__ == '__main__':
    config = model_config()
    run_type = config.run_type
    if run_type == "train":
        train(config)
    elif run_type == "test":
        cross_domain(config.save_dir + config.target_domain+'/model', config.device)

