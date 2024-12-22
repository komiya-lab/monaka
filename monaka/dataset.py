# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import namedtuple

import numpy as np
import torch
import torch.distributed as dist
from typing import Union, List, Dict, Optional
from monaka.tokenizer import Tokenizer
from monaka.mylogging import logger
from transformers import  AutoTokenizer



class LUWJsonLDataset(torch.utils.data.Dataset):
    r"""
    JsonL形式のデータセット 各行は以下。
    {
        "sentence": str,   # 文そのもの
        "tokens": [str, ]  # 短単位のリスト
        "pos": [str, ]     # 形態論情報(短単位)
        "lemma": [str,]    # 語彙素原形(必須フィールドではない)
        "labels": [str, ]  # ラベル。短単位ごとに付与
    }

    Args:
        jsonlfiles (str or list[str]):
            読み込み対象のファイル名（のリスト)
        label_file (str):
            全ラベルを記載したjsonファイル label: id 形式。未知ラベルは常に unk: 0
        pos_file (str):
            形態素IDを記載したJSONファイル label: id 形式。未知ラベルは常に unk: 0 pos_as_tokensの時は利用されない。
        lm_tokenizer (str):
            適切なTransoformesのTokenizerをラップしたTokenizer名
        lm_tokenizer_config (dict):
            Tokenizerのconfig
        max_length (int):
            データの最大長
        pos_as_tokens (bool):
            形態論情報を短単位の後に付与するかどうか default=False
        label_for_all_subwords (bool):
            サブワード単位でラベル付けをする。default False
        kwargs (dict):
            Keyword arguments that will be passed into :meth:`transform.load` together with `data`
            to control the loading behaviour.

    Attributes:
        sentences (list[Sentence]):
            A list of sentences loaded from the data.
            Each sentence includes fields obeying the data format defined in ``transform``.
    """

    def __init__(self, jsonlfiles: Union[str, List[str]], label_file: str, pos_file: str, lm_tokenizer: str, lm_tokenizer_config: Dict, max_length: int=1024, pos_as_tokens: bool=False, 
                 label_for_all_subwords: bool=False, logger=logger, store_all: bool=False, 
                 **kwargs):
        self.sentences = list()
        self.tokenizer = Tokenizer.by_name(lm_tokenizer)(**lm_tokenizer_config)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pos_as_tokens = pos_as_tokens
        self.label_for_all_subwords = label_for_all_subwords
        self.max_length = max_length
        self.jsonlfiles = jsonlfiles
        self.logger = logger
        self.store_all = store_all

        with open(label_file) as f:
            self.label_dic = json.load(f)

        if pos_file is not None and not pos_as_tokens:
            with open(pos_file) as f:
                self.pos_dic = json.load(f)
        else:
            self.pos_dic = None
        
        if isinstance(jsonlfiles, (str, Path)):
            self.logger.info(f"loading {jsonlfiles}")
            self.load(jsonlfiles)
        elif isinstance(jsonlfiles, list):
            for fname in jsonlfiles:
                if isinstance(fname, str):
                    self.logger.info(f"loading {fname}")
                    self.load(fname)
                elif isinstance(fname, dict):
                    self.load_dict(fname)
        
        self.logger.info(f"total {len(self.sentences)} sentences loaded.")
        super().__init__()

    @staticmethod
    def collate_function(data: List[Dict]):
        #targets = ["input_ids", "label_ids", "pos_ids"]
        res = dict()
        #for target in targets:
        #    if target not in data[0]:
        #        continue
        #    res[target] = [d[target] for d in data]
        for k in data[0].keys():
            res[k] = [d.get(k) for d in data]
        return res

    def load(self, jsonlfile: str):
        with open(jsonlfile) as f:
            for line in f:
                js = json.loads(line)
                self.load_dict(js)

    def load_dict(self, js: dict):
        js['skip'] = False
        if len(js["pos"]) != len(js["tokens"]):
            self.logger.warn(f'skip loading {js["sentence"]} because of pos {len(js["pos"])} and token {len(js["tokens"])} length unmatch')
            if self.store_all:
                js['skip'] = True
                self.sentences.append(js)
            return
        if len(js["pos"]) == 0:
            self.logger.warn(f'skip loading {js["sentence"]} because there is no token.')
            if self.store_all:
                js['skip'] = True
                self.sentences.append(js)
            return

        js["subwords"] = self.to_token_ids(js["tokens"], js["pos"] if self.pos_as_tokens else None)
        js["input_ids"] = torch.LongTensor(js["subwords"]["input_ids"])
        js["label_ids"] = self.to_label_ids(js["labels"], js["subwords"].word_ids() if self.label_for_all_subwords else None) if "labels" in js else None

        if 'input_ids' not in js:
            return

            #for l_subs, lid in zip(js["lemma_subwords"].word_ids(), js["lemma_id"]):
            #    js["lemma_target"].extend([lid for _ in range(len(l_subs))])
        
            #print(len(js["lemma_target"] ), len(js["input_ids"] ))

        if self.pos_dic:
            js["pos_ids"] = self.to_pos_ids(js["pos"], js["subwords"].word_ids() if self.label_for_all_subwords else None) 


        if len(js["subwords"].word_ids()) == 0:
            self.logger.warning(f"no words: {js['tokens']}")
            if self.store_all:
                js['skip'] = True
                self.sentences.append(js)
            return

        if len(js["pos"]) != np.max(js["subwords"].word_ids()) + 1:
            self.logger.warning(f'unmatch length {len(js["pos"])} {np.max(js["subwords"].word_ids()) + 1}, {js["tokens"]} {js["subwords"]} {js["subwords"].word_ids()}')
        
        if "lemma" in js:
            js["lemma_subwords"] = self.to_token_ids(js["lemma"])
            js["lemma_ids"] = []
            #torch.LongTensor(js["lemma_subwords"]["input_ids"]) # lemma size list of subword list
            sub_idx = js["lemma_subwords"]["input_ids"]
            w_idx = js["subwords"].word_ids()
            l_idx = js["lemma_subwords"].word_ids()
            if len(l_idx) == 0:
                self.logger.warn(f"no lemma: {js['lemma']}")
                self.sentences.append(js)
                return

            for i in range(max(l_idx) + 1):
                js["lemma_ids"].append(torch.LongTensor([sid for sid, wid in zip(sub_idx, l_idx) if wid == i]))
            
            js["lemma_target"] = [js["lemma_id"][i] for i in w_idx]
        
            #print(len(js["lemma_target"] ), len(js["input_ids"] ))

        if len(js["pos"]) != np.max(js["subwords"].word_ids()) + 1:
            self.logger.warning(f'unmatch length {len(js["pos"])} {np.max(js["subwords"].word_ids()) + 1}, {js["tokens"]} {js["subwords"]} {js["subwords"].word_ids()}')
        self.sentences.append(js)

    def to_label_ids(self, labels: List[str], word_ids: Optional[List[int]]=None):
        labels_ = [self.label_dic.get(k, 0) for k in labels]
        if word_ids is not None:
            prv = -1
            labels = list()
            for idx in word_ids:
                if idx != prv:
                    labels.append(labels_[idx])
                    prv = idx
                else:
                    labels.append(1)
        else:
            labels = labels_
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        return torch.LongTensor(labels)
    

    def to_pos_ids(self, labels: List[str], word_ids: Optional[List[int]]=None):
        labels_ = [self.pos_dic.get(k, 0) for k in labels]
        if word_ids is not None:
            prv = -1
            labels = list()
            for idx in word_ids:
                if idx != prv:
                    labels.append(labels_[idx])
                    prv = idx
                else:
                    labels.append(1) # padding index = 1
        else:
            labels = labels_
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        return torch.LongTensor(labels)
    
    def to_token_ids(self, tokens: List[str], pos: Optional[List[str]]=None):
        if pos is not None:
            assert(len(tokens) == len(pos))
            targets = [f"{self.replace_token(t)} {p}" for t, p in zip(tokens, pos)]
        else:
            targets = [self.replace_token(t) for t in tokens]

        return self.tokenizer.tokenize(targets, max_length=self.max_length)

    @staticmethod
    def replace_token(token: str) -> str:
        """うまくtokenizeできない句点記号を置換する"""
        if token in ["．", "："]:
            return "。"
        if token in ["，", "；"]:
            return "、"
        if token in ["？"]:
            return "?"
        if token in ["！"]:
            return "!"
        if token in ["（"]:
            return "("
        if token in ["）"]:
            return ")"
        return token

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"

        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]


class LemmaJsonDataset(torch.utils.data.Dataset):
    r"""
    Json形式のデータセット 
    "key (全情報)": {
        "surface": "やまとうた", # 表層形
        "lemma": "やまと歌",  #  長単位原形
        "pos": "名詞-普通名詞-一般", # 品詞
        "suw": [
        {
            "pos": "名詞-固有名詞-地名-一般", # 短単位品詞
            "lemma": "ヤマト", # 短単位原形
            "surface": "やまと"# 短単位原形
        },
        {
            "pos": "名詞-普通名詞-一般", # 短単位品詞
            "lemma": "歌", # 短単位原形
            "surface": "うた" # 短単位原形
        }
        ],
        "input": "pos: 名詞-普通名詞-一般 surface: やまとうた suw_pos: 名詞-固有名詞-地名-一般 名詞-普通名詞-一般 suw_surface: やまと うた suw_lemma: ヤマト 歌" #入力形式
    }

    Args:
        jsonfiles (str or list[str]):
            読み込み対象のファイル名（のリスト)
        transformer_name (str):
            Transformers.from_pretrainedで読めるもの
        fields (List[str]):
            指定したフィールドをjsonから取得し、入力に変換する
        kwargs (dict):
            Keyword arguments that will be passed into :meth:`transform.load` together with `data`
            to control the loading behaviour.

    Attributes:
    """
    def __init__(self, jsonfiles: Union[str, List[Union[str, Dict]]], transformer_name: str, max_length: int=512, fields: Optional[List[str]]=None, logger=logger,
                 **kwargs):
        
        self.lemma = list()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        if isinstance(jsonfiles, str):
            self.load_data(jsonfiles, fields)
        else:
            for jsfile in jsonfiles:
                if isinstance(jsfile, str):
                    self.load_data(jsfile, fields)
                else:
                    self.load_dict(jsfile, fields)

    def __getitem__(self, idx):
        if np.random.random() < 0.1:
            t = f"input: {self.lemma[idx]['input']} target: {self.lemma[idx]['target']}"
            #print(t)
            logger.info(t)
        return self.lemma[idx]
    
    def __len__(self):
        return len(self.lemma)

    @staticmethod
    def to_label(data: Dict, fields: Optional[List[str]]):
        if fields is None:
            return data["input"]
        
        inputs: List[str] = list()
        for field in fields:
            if not field.startswith("suw_"):
                t = f"{field}: {data.get(field, '')}"
                inputs.append(t)
                continue

            starget = field.split("_")[1]
            vals = [suw[starget] for suw in data["suw"]]
            inputs.append(f'{field}: {" ".join(vals)}')
        
        return " ".join(inputs)
    
    def load_data(self, fname: str, fields: Optional[List[str]]):
        with open(fname) as f:
            data: Dict = json.load(f)

        self.load_dict(data, fields)

    def load_dict(self, data: Dict, fields: Optional[List[str]]):
        for val in data.values():
            d = {
                "input": self.to_label(val, fields),
                "target": val["lemma"] if 'lemma' in val else None
            }
            d["subwords"] = self.tokenizer(d["input"], max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
            d["input_ids"] =torch.LongTensor(d["subwords"]["input_ids"][0])
            d["attention_mask"] = torch.tensor(d["subwords"]["attention_mask"])
            d["target_subwords"] = self.tokenizer(d["target"], max_length=self.max_length, padding='max_length') if d['target'] is not None else None
            if d['target'] is not None:
                d["labels"] = d["target_subwords"]["input_ids"]
            #d["decoder_input_ids"] = d["target_subwords"]["input_ids"]
            self.lemma.append(d)
        
