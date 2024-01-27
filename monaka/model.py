# -*- coding: utf-8 -*-

import os
import json
from random import Random

import torch
import torch.nn as nn
import torch.distributed as dist
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

    def forward(self, words: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self, out, labels, mask) -> torch.Tensor:
        raise NotImplementedError
    

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

    def forward(self, words: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
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
    
    def loss(self, out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """
        out: [batch, words_len, n_class]
        labels: [batch, 1]
        mask: mask
        """

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
