# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


# The code is based on repository: https://github.com/yangheng95/LCF-ABSA


import torch
import torch.nn as nn
import copy
import numpy as np
from layers.attention import NoQueryAttention
from pytorch_transformers.modeling_bert import *


def init_params(model, initializer):
    for child in model.children():
        # skip bert params (with unfreezed bert)
        if type(child) != BertModel and type(child) != nn.Embedding:
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)


class SelfAttention(nn.Module):
    def __init__(self, config, L):
        super(SelfAttention, self).__init__()
        self.SA = BertSelfAttention(config)
        self.L = L
        self.pooler = BertPooler(config)

    def forward(self, inputs):
        SA_out = self.SA(inputs, 0.0)
        out = self.pooler(torch.tanh(SA_out[0]))
        return out


class VHBertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):
        super(VHBertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = attention_mask

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        ## sequence_output, pooled_output, (hidden_states), (attentions)
        return {'sequence_output': sequence_output}


class VH_BERT_TSA(nn.Module):
    def __init__(self, config='a class with num_attention_heads, hidden_size, attention_probs_dropout_prob, output_attentions',
                 bert_dir='/mnt/sda1/bert/uncased_L-12_H-768_A-12',
                 drop=0.0,
                 head_type_vec=[1] * 3 + [0] * 9,
                 L=80, win_size=5,
                 bert_dim=768, num_class=3
                 ):
        super(VH_BERT_TSA, self).__init__()
        self.register_buffer('att_mask', self.get_atten_mask(
            head_type_vec, L, win_size))
        self.text_bert = VHBertModel.from_pretrained(bert_dir)
        self.aspect_bert = BertModel.from_pretrained(bert_dir)
        self.aspect_self_att_pooler = SelfAttention(config, L)
        self.aspect_emphasize_att = NoQueryAttention(
            embed_dim=bert_dim * 2, hidden_dim=bert_dim, out_dim=bert_dim)
        self.reduce2_bert_dim_linear = nn.Linear(bert_dim*2, bert_dim)
        self.reduce2_num_class_linear = nn.Linear(bert_dim, num_class)
        self.drop = drop
        self.L = L

    def get_atten_mask(self, head_type_vec, L, win_size):
        # num_head L L
        att_mask = []
        local_mask_map = self.get_local_mask_map(L, win_size)
        for tp in head_type_vec:
            if tp == 0:
                att_mask.append(torch.ones(L, L))
            else:
                att_mask.append(local_mask_map)
        return torch.stack(att_mask)

    def get_local_mask_map(self, L, win_size):
        assert win_size % 2 == 1, 'win_size should be odd'
        half = win_size // 2
        mask = torch.ones(L, L)
        for i in range(L):
            left = max(0, i - half)
            right = min(L, i+half+1)
            mask[i][left:right] = torch.zeros(right - left)
        return mask

    def forward(self, input_ids, attention_mask, pos):
        # self.att_mask: num_head L L--> 1 num_head L L, attention_mask:B L-->B 1 1 L
        vh_attention_mask = self.att_mask.unsqueeze(
            0) * attention_mask.unsqueeze(1).unsqueeze(2).float()
        text = torch.tanh(self.text_bert(input_ids, attention_mask=self.att_mask)[
            'sequence_output'])
        aspect = self.aspect_bert(input_ids, attention_mask=attention_mask)[0]
        # B L
        mask = torch.zeros_like(input_ids).float()
        for i, (s, e) in enumerate(pos):
            mask[i, s:e] = torch.tensor([1]*(e.item()-s.item()))
        aspect = aspect * mask.unsqueeze(-1)  # B L H
        aspect_rep = self.aspect_self_att_pooler(aspect)  # B H

        cat = torch.cat(
            [text, aspect_rep.unsqueeze(1).expand(-1, self.L, -1)], -1)
        cat = torch.nn.functional.dropout(cat, self.drop)  # B L H

        aspect_emphasized_text_rep = self.aspect_emphasize_att(cat)[
            'output']  # B 1 H
        aspect_emphasize_att_score = self.aspect_emphasize_att(cat)[
            'att_score']  # B h 1 L
        # aspect_emphasized_text_rep = torch.tanh(aspect_emphasized_text_rep)

        concat = torch.cat(
            [aspect_emphasized_text_rep.squeeze(), aspect_rep], -1)  # B 2H
        concat = torch.nn.functional.dropout(concat, self.drop)

        hidden = self.reduce2_bert_dim_linear(concat)
        hidden = torch.nn.functional.relu(hidden)
        hidden = torch.nn.functional.dropout(hidden, self.drop)
        out = self.reduce2_num_class_linear(hidden)
        return {'output': out, 'aspect_emphasize_att_acore_BhL': aspect_emphasize_att_score.squeeze().detach().cpu().numpy()}


if __name__ == "__main__":
    from types import SimpleNamespace
    conf = SimpleNamespace(num_attention_heads=12, hidden_size=768,
                           attention_probs_dropout_prob=0.1, output_attentions=False)
    m = VH_BERT_TSA(conf)
    input_ids = torch.ones((16, 28))
    pos = torch.tensor([[1, 4] for _ in range(16)])
    print(m.forward(input_ids, pos)['output'].size())
