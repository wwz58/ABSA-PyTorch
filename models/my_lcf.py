# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


# The code is based on repository: https://github.com/yangheng95/LCF-ABSA


import torch
import torch.nn as nn
import copy
import numpy as np
from pytorch_transformers.modeling_bert import BertModel, BertPooler, BertSelfAttention, BertConfig


class SelfAttention(nn.Module):
    def __init__(self, config, L):
        super(SelfAttention, self).__init__()
        self.SA = BertSelfAttention(config)
        self.L = L
        self.pooler = BertPooler(config)

    def forward(self, inputs):
        SA_out = self.SA(inputs, 0.0)
        out = torch.tanh(SA_out[0])
        return out


class MY_BERT_LCF(nn.Module):
    def __init__(self, config='a class with num_attention_heads, hidden_size, attention_probs_dropout_prob, output_attentions',
                 bert_dir='/mnt/sda1/bert/uncased_L-12_H-768_A-12',
                 drop=0.0, L=80,
                 bert_dim=768, num_class=3,
                 SDR=5, tp='cdm'
                 ):
        super(MY_BERT_LCF, self).__init__()
        self.text_bert = BertModel.from_pretrained(bert_dir)
        self.aspect_bert = copy.deepcopy(self.text_bert)
        self.aspect_self_att = SelfAttention(config, L)
        self.bert_pooler = BertPooler(config)
        if tp == 'cdm':
            self.reduce2_bert_dim = nn.Linear(bert_dim * 2, bert_dim)
        self.reduce2_num_class_linear = nn.Linear(bert_dim, num_class)
        self.drop = drop
        self.L = L
        self.SDR = SDR
        self.tp = tp

    def forward(self, text_asp_ids, text_asp_att_mask, text_ids, text_att_mask, pos):
        text_asp = torch.tanh(self.text_bert(
            text_asp_ids, attention_mask=text_asp_att_mask)[0])
        aspect = self.aspect_bert(text_ids, attention_mask=text_att_mask)[0]
        if self.tp == 'cdm':
            mask = torch.zeros_like(text_ids).float()  # B L
            for i, (s, e) in enumerate(pos):
                s = max(0, s.item() - self.SDR)
                e = min(e.item()+self.SDR+1, len(aspect))
                mask[i, s:e] = torch.tensor([1.0] * (e - s))
            aspect = aspect * mask.unsqueeze(-1)
        cat = torch.cat([text_asp, aspect], -1)  # B L 2H
        cat = self.reduce2_bert_dim(cat)
        x = self.aspect_self_att(cat)  # B L H
        x = self.bert_pooler(cat)  # B H
        out = self.reduce2_num_class_linear(x)
        # , 'aspect_emphasize_att_acore_BhL': aspect_emphasize_att_score.squeeze().detach().cpu().numpy()}
        return {'output': out}


if __name__ == "__main__":
    conf = BertConfig.from_pretrained('/mnt/sda1/bert/uncased_L-12_H-768_A-12')
    m = MY_BERT_LCF(conf)
    input_ids = torch.ones((16, 80)).long()
    attention_mask = torch.ones_like(input_ids)

    text_asp_ids, text_asp_att_mask = copy.deepcopy(
        input_ids), copy.deepcopy(attention_mask)

    pos = torch.LongTensor([[1, 4] for _ in range(16)])
    print(m.forward(text_asp_ids, text_asp_att_mask, input_ids,
                    attention_mask, pos)['output'].size())
