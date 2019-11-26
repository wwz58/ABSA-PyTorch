# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy

from pytorch_transformers import BertModel, BertConfig, BertTokenizer

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset

from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from models.att_att import ATT_ATT
from models.vh_bert import VH_BERT_TSA
from models.bert_att import BERT_ATT
from models.my_lcf import MY_BERT_LCF

from tensorboardX import SummaryWriter
import pandas as pd
import pickle

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if opt.model_name.lower() in ['vh_bert', 'bert_att', 'my_lcf']:
            tokenizer = BertTokenizer.from_pretrained(
                opt.pretrained_bert_name)
            config = BertConfig.from_pretrained(
                opt.pretrained_bert_name, output_attentions=True)
            self.model = opt.model_class(config, ).to(opt.device)
        elif 'bert' in opt.model_name.lower():
            tokenizer = Tokenizer4Bert(
                opt.max_seq_len, opt.pretrained_bert_name)
            config = BertConfig.from_pretrained(
                opt.pretrained_bert_name, output_attentions=True)
            bert = BertModel.from_pretrained(
                opt.pretrained_bert_name, config=config)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='./cache/{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='./cache/{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(
                self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(
                torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(
            n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel and type(child) != torch.nn.Embedding:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, save_dir, writer):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(
                    self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)

                output_dic = self.model(*inputs)
                outputs = output_dic['output']

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1)
                              == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(
                        train_loss, train_acc))
                    writer.add_scalar('tr_loss', train_loss, global_step)
                    writer.add_scalar('tr_acc', train_acc, global_step)

            val_acc, val_f1, pred_logits, _ = self._evaluate_acc_f1(
                val_data_loader)
            writer.add_scalar('dev_acc', val_acc, epoch)
            writer.add_scalar('dev_f1', val_f1, epoch)
            logger.info(
                '> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                preds = pred_logits.detach().max(-1)[1].tolist()
                pd.DataFrame(preds, columns=['y_pred']).to_csv(
                    os.path.join(save_dir, 'dev_y_pred.txt'), header=None, index=False)
                path = os.path.join(save_dir, 'best.pt')
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        t_att_all = []  # total_B n_head q_len k_len
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(
                    self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)

                output_dic = self.model(*t_inputs)
                t_outputs = output_dic['output']
                att_score = output_dic.get(
                    'aspect_emphasize_att_acore_BhL', None)

                n_correct += (torch.argmax(t_outputs, -1)
                              == t_targets).sum().item()
                n_total += len(t_outputs)

                t_att_all.append(att_score)
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat(
                        (t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat(
                        (t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(
            t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1, t_outputs_all, t_att_all

    def run(self):
        save_dir = os.path.join(
            'out', f'{self.opt.model_name}-{self.opt.dataset}-{self.opt.exp_id}')
        os.makedirs(save_dir, exist_ok=True)
        run_dir = os.path.join(
            'runs', f'{self.opt.model_name}-{self.opt.dataset}-{self.opt.exp_id}')
        writer = SummaryWriter(run_dir)
        log_file = os.path.join(save_dir, 'log.txt')
        logger.addHandler(logging.FileHandler(log_file))

        if self.opt.do_train:
            criterion = nn.CrossEntropyLoss()
            _params = filter(lambda p: p.requires_grad,
                             self.model.parameters())
            optimizer = self.opt.optimizer(
                _params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

            train_data_loader = DataLoader(
                dataset=self.trainset, batch_size=self.opt.tr_batch_size, shuffle=True)

            val_data_loader = DataLoader(
                dataset=self.valset, batch_size=self.opt.te_batch_size, shuffle=False)

            self._reset_params()
            best_model_path = self._train(
                criterion, optimizer, train_data_loader, val_data_loader, save_dir, writer)

        if self.opt.do_eval_train:
            if self.opt.do_train:
                test_data_loader = train_data_loader
            else:
                best_model_path = os.path.join(save_dir, 'best.pt')
                test_data_loader = DataLoader(
                    dataset=self.trainset, batch_size=self.opt.te_batch_size, shuffle=True)
            self.model.load_state_dict(torch.load(best_model_path))
            self.model.eval()
            test_acc, test_f1, test_pred_logits, te_att_all = self._evaluate_acc_f1(
                test_data_loader)
            with open(os.path.join(save_dir, 'train_att_score.pkl'), 'wb') as f:
                pickle.dump(te_att_all, f)
            test_preds = test_pred_logits.detach().max(-1)[1].tolist()
            pd.DataFrame(test_preds, columns=['y_pred']).to_csv(
                os.path.join(save_dir, 'train_y_pred.txt'), header=None, index=False)
            logger.info('>> train_acc: {:.4f}, train_f1: {:.4f}'.format(
                test_acc, test_f1))
            writer.add_text(
                'train_report', f'train_acc: {test_acc:.4f}, train_f1: {test_f1:.4f}', 0)

        if self.opt.do_eval:
            test_data_loader = DataLoader(
                dataset=self.testset, batch_size=self.opt.te_batch_size, shuffle=False)
            if not self.opt.do_train:
                best_model_path = os.path.join(save_dir, 'best.pt')
            self.model.load_state_dict(torch.load(best_model_path))
            self.model.eval()
            test_acc, test_f1, test_pred_logits, test_att_score = self._evaluate_acc_f1(
                test_data_loader)
            with open(os.path.join(save_dir, 'test_a_c_att.pkl'), 'wb') as f:
                pickle.dump(test_att_score, f)
            test_preds = test_pred_logits.detach().max(-1)[1].tolist()
            pd.DataFrame(test_preds, columns=['y_pred']).to_csv(
                os.path.join(save_dir, 'test_y_pred.txt'), header=None, index=False)
            logger.info('##'*20)
            logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(
                test_acc, test_f1))
            logger.info('##'*20)

            writer.add_text(
                'test_report', f'test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}', 0)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', default=0, type=int)
    parser.add_argument('--model_name', default='bert_att', type=str)
    parser.add_argument('--dataset', default='laptop',
                        type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--do_train', action='store_true',
                        help='do_train')
    parser.add_argument('--do_eval', action='store_true',
                        help='do_eval')
    parser.add_argument('--do_eval_train', action='store_true',
                        help='do_eval_train')
    parser.add_argument('--max_seq_len', default=80, type=int)

    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=5, type=int,
                        help='try larger number for non-BERT models')
    parser.add_argument('--tr_batch_size', default=16, type=int,
                        help='tr try 16, 32, 64 for BERT models')
    parser.add_argument('--te_batch_size', default=16, type=int,
                        help='te try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)

    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--hops', default=3, type=int)

    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name',
                        default='/mnt/sda1/bert/uncased_L-12_H-768_A-12', type=str)
    parser.add_argument('--polarities_dim', default=3, type=int)

    parser.add_argument('--device', default='cuda:0',
                        type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=2019, type=int,
                        help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')

    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm',
                        type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=5, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'my_lcf': MY_BERT_LCF,
        'bert_att': BERT_ATT,
        'vh_bert': VH_BERT_TSA,
        'att_att': ATT_ATT,
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    input_colses = {
        'my_lcf': ['text_asp_ids', 'text_asp_att_mask', 'input_ids', 'attention_mask', 'pos'],
        'bert_att': ['text_asp_ids', 'text_asp_att_mask', 'input_ids', 'attention_mask', 'pos'],
        'vh_bert': ['input_ids', 'attention_mask', 'pos'],
        'att_att': ['text_raw_indices', 'aspect_indices'],
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
