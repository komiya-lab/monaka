# -*- coding: utf-8 -*-

import os
import sys
import json
from random import Random

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from registrable import Registrable
from typing import Dict
from monaka.module import MLP, LMEmbedding
from monaka.mylogging import logger


class LUWParserModel(nn.Module, Registrable):

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        Registrable.__init__(self)

    @classmethod
    def from_config(cls, config: Dict, label_file: str, pos_file: str, **kwargs):
        with open(label_file) as f:
            js = json.load(f)
            config["n_class"] = len(js)

        
        with open(pos_file) as f:
            js = json.load(f)
            config["n_pos"] = len(js)

        return cls(**config)

    def forward(self, words: torch.Tensor, word_ids: torch.Tensor, pos: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self, out, labels, mask, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
class LUWLemmaModel(nn.Module, Registrable):
    """
    語彙素原形を推定するモデルの基底クラス
    """

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        Registrable.__init__(self)

    @classmethod
    def from_config(cls, config: Dict,  **kwargs):

        return cls(**config)

    def forward(self, words: torch.Tensor, luw_target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self, out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    

@LUWLemmaModel.register("FixLen")
class FixLenLemmaModel(LUWLemmaModel):
    """
    固定長のサブワードで原形を推定するモデル

    Args:
        lm_class_name (str):
            用いるlm class名 TrasformersのAutoConfig, AutoModelなどが上手く使えない場合は専用クラスが用意されている。
        lm_class_config (dict):
            lm_class用のconfig
        max_len: (int)
            原形のサブワード最大長
        dropout: (float)
            dropout
    """

    def __init__(self,
            lm_class_name: str,
            lm_class_config: Dict,
            max_len: int,
            dropout: float, 
            **kwargs) -> None:
        super().__init__(**kwargs)

        self.max_len = max_len
        self.dropout = dropout
        self.lm_class_name = lm_class_name
        self.lm_class_config = lm_class_config

        self.m_lm = LMEmbedding.by_name(lm_class_name)(**lm_class_config)
        self.m_out = MLP(self.m_lm.n_out, self.m_lm.n_vocab) # 埋め込み表現次元 -> subword ID

        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, inputs: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        """
        inputs: [batch, subwords_len]
        target: [batch, subword_len] the indices pointing to target luw
        """
        words_emb = self.m_lm(inputs)
        L = torch.max(target) + 1
        embs = list()

        for i in range(L):
            wi = target.eq(i).unsqueeze(-1)
            l = torch.sum(wi)
            mask = torch.cat([wi for _ in range(words_emb.size()[-1])], dim=-1) # batch, n subwords, hidden
            #print(words_emb.size())
            #print(mask.size())
            #_o = words_emb * mask # batch (actually = 1), num of subwords in a word, hidden (acctually masked zero)
            _o = words_emb[mask].reshape((l, words_emb.size()[-1]))
            if l > self.max_len:
                _o = _o[:self.max_len, :]
            embs.append(_o)

        # サイズを維持するためにわざと追加
        temp_embs = words_emb[0, :self.max_len, :]
        embs.append(temp_embs)
            

        embs = pad_sequence(embs, batch_first=True) # target luw + 1, self.max_len, hidden
        #print(embs.size())
        embs = embs[:-1, :, :] # target luw, self.max_len, hidden
        #print(embs.size())
        out = self.m_out(embs)

        return out

    def loss(self, out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        """
        out: [target luw, self.max_len, n_class]
        labels: [target luw, max lemma, 1]
        mask: mask
        """

        osize = out.size()
        labels_ = torch.zeros((osize[0], osize[1], 1), device=out.device, dtype=torch.long)
        lsize = labels.size()
        labels_[:lsize[0], :lsize[1], :] = labels.unsqueeze(-1)
        #print(out.size(), labels_.size())
        return self.criterion(out.flatten(0, 1), labels_.flatten()) 

@LUWParserModel.register("SeqTagging")
class SeqTaggingParserModel(LUWParserModel):
    """
    SubwordレベルでSequence Taggingするモデル

    Args:
        n_pos (int):
            形態素の種類数。
        n_pos_emb (int):
            形態素埋め込み表現の次元数。
        n_class (int):
            クラスラベル数
        pos_dropout (float):
            pos埋め込みのdropout
        mlp_dropout (float):
            識別用のMLPのdropout
        lm_class_name (str):
            用いるlm class名 TrasformersのAutoConfig, AutoModelなどが上手く使えない場合は専用クラスが用意されている。
        lm_class_config (dict):
            lm_class用のconfig
        pos_padding_idx (int):
            pos埋め込みのpadding idx
    """
    
    def __init__(self,
            n_pos: int,
            n_pos_emb: int,
            n_class: int,
            pos_dropout: float,
            mlp_dropout: float,
            lm_class_name: str,
            lm_class_config: Dict,
            pos_padding_idx: int = 1,
            **kwargs) -> None:
        super().__init__(**kwargs)
        
        logger.info("Model: SeqTagging")

        self.n_pos = n_pos
        self.n_pos_emb = n_pos_emb
        self.n_class = n_class
        self.pos_dropout = pos_dropout
        self.lm_class_name = lm_class_name
        self.lm_class_config = lm_class_config
        self.pos_padding_idx = pos_padding_idx
        
        self.m_lm = LMEmbedding.by_name(lm_class_name)(**lm_class_config)
        self.m_pos_emb = nn.Embedding(n_pos, n_pos_emb, pos_padding_idx) if n_pos > 0 and n_pos_emb > 0 else None
        self.m_pos_dropout = nn.Dropout(pos_dropout) if self.m_pos_emb else None

        self.n_in = self.m_lm.n_out if self.m_pos_emb is None else self.m_lm.n_out + n_pos_emb
        self.m_out = MLP(self.n_in, n_class, mlp_dropout)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words: torch.Tensor, word_ids: torch.Tensor, pos: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        words: [batch, words_len]
        pos: [batch, owrds_len]
        """
        #print(words.size())
        words_emb = self.m_lm(words)
        if self.m_pos_emb:
            #print(pos)
            pos_embs = self.m_pos_emb(pos)
            pos_embs = self.m_pos_dropout(pos_embs)
            #print(words_emb.size())
            #print(pos_embs.size())
            words_emb = torch.cat((words_emb, pos_embs), dim=-1) # batch, words_len, hidden

        out = self.m_out(words_emb)

        return out # batch, words_len, n_class
    
    def loss(self, out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        """
        out: [batch, words_len, n_class]
        labels: [batch, 1]
        mask: mask
        """

        return self.criterion(out[mask], labels[mask])



@LUWParserModel.register("WordTagging")
class WordTaggingParserModel(LUWParserModel):
    """
    WordレベルでSequence Taggingするモデル

    Args:
        n_pos (int):
            形態素の種類数。
        n_pos_emb (int):
            形態素埋め込み表現の次元数。
        n_class (int):
            クラスラベル数
        pooling (str):
            subword -> wordのpooling方法 (max, sum, attention)
        pos_dropout (float):
            pos埋め込みのdropout
        mlp_dropout (float):
            識別用のMLPのdropout
        lm_class_name (str):
            用いるlm class名 TrasformersのAutoConfig, AutoModelなどが上手く使えない場合は専用クラスが用意されている。
        lm_class_config (dict):
            lm_class用のconfig
        pos_padding_idx (int):
            pos埋め込みのpadding idx
    """
    
    def __init__(self,
            n_pos: int,
            n_pos_emb: int,
            n_class: int,
            pooling: str,
            pos_dropout: float,
            mlp_dropout: float,
            lm_class_name: str,
            lm_class_config: Dict,
            pos_padding_idx: int = 1,
            **kwargs) -> None:
        super().__init__(**kwargs)
        
        logger.info("Model: WordTagging")

        self.n_pos = n_pos
        self.n_pos_emb = n_pos_emb
        self.n_class = n_class
        self.pooling_name = pooling
        self.pos_dropout = pos_dropout
        self.lm_class_name = lm_class_name
        self.lm_class_config = lm_class_config
        self.pos_padding_idx = pos_padding_idx
        
        self.m_lm = LMEmbedding.by_name(lm_class_name)(**lm_class_config)
        self.m_pos_emb = nn.Embedding(n_pos, n_pos_emb, pos_padding_idx) if n_pos > 0 and n_pos_emb > 0 else None
        self.m_pos_dropout = nn.Dropout(pos_dropout) if self.m_pos_emb else None

        if "max" in pooling:
            self.pooling = self.max
        elif "sum" in pooling:
            self.pooling = torch.sum
        else:
            self.pooling = None

        self.n_in = self.m_lm.n_out if self.m_pos_emb is None else self.m_lm.n_out + n_pos_emb
        self.m_out = MLP(self.n_in, n_class, mlp_dropout)

        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def max(value, **kwargs):
        return torch.max(value, **kwargs).values

    def forward(self, words: torch.Tensor, word_ids: torch.Tensor, pos: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        words: [batch, subwords_len]
        word_ids: [batch, subword_len] the indices pointing to original words
        pos: [batch, words_len]
        """
        #print(words.size())
        words_emb = self.m_lm(words)

        we = list()
        if self.m_pos_emb and torch.max(word_ids)+1 != pos.size()[-1]:
            print(torch.max(word_ids, dim=1), file=sys.stderr)
        L = torch.max(word_ids) + 1 if not self.m_pos_emb else pos.size()[-1] # なぜかPOSが多い時がある。調査要

        for i in range(L):
            wi = word_ids.eq(i).unsqueeze(-1)
            mask = torch.cat([wi for _ in range(words_emb.size()[-1])], dim=-1)
            #print(words_emb.size())
            #print(mask.size())
            _o = words_emb * mask # batch, num of subwords in a word, hidden (acctually masked zero)
            we.append(self.pooling(_o, dim=1, keepdim=True)) # batch, 1, hidden

        words_emb = torch.cat(we, 1)

        if self.m_pos_emb:
            #print(pos)
            pos_embs = self.m_pos_emb(pos)
            pos_embs = self.m_pos_dropout(pos_embs)
            #print(words_emb.size())
            #print(pos_embs.size())
            words_emb = torch.cat((words_emb, pos_embs), dim=-1) # batch, words_len, hidden

        out = self.m_out(words_emb) # batch, len, hidden
        return out # batch, words_len, n_class
    
    def loss(self, out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        """
        out: [batch, words_len, n_class]
        labels: [batch, 1]
        mask: mask
        """
        out_size = out.size()
        mask = mask[:out_size[0], :out_size[1]]
        labels = labels[:out_size[0], :out_size[1]]

        return self.criterion(out[mask], labels[mask])



class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)

    def __getattr__(self, name):
        wrapped = super().__getattr__('module')
        if hasattr(wrapped, name):
            return getattr(wrapped, name)
        return super().__getattr__(name)


def init_device(device, backend='nccl', host=None, port=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    if torch.cuda.device_count() > 1:
        host = host or os.environ.get('MASTER_ADDR', 'localhost')
        port = port or os.environ.get('MASTER_PORT', str(Random(0).randint(10000, 20000)))
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = port
        dist.init_process_group(backend)
        torch.cuda.set_device(dist.get_rank())


def is_master():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
