# -*- coding: utf-8 -*-

import os
import glob
import json
import torch
import fugashi

from typing import List, Dict, Any, Optional
from registrable import Registrable

from monaka.model import LUWParserModel, init_device, is_master


class Decoder(Registrable):

    def __init__(self) -> None:
        super().__init__()


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.decode(*args, **kwds)

    
    def decode(self, tokens: List[str], pos: List[str], labels: List[str], **kwargs) -> Dict:
        """
        出力は辞書形式 LUWは開始位置の場合はPOS-tag名、そうでない場合は"*"。文節は開始位置は"B"そうでなければ"I"。
        解析対象のfieldを含んでいれば良い。
        kwargsにメタ情報を追記でき、それらを辞書に加えることを想定している
        {
            "luw": ["POS-tag or *"]
            "chunk": ["B", "I"]
        }
        """
        raise NotImplementedError


@Decoder.register("LUW-Bunsetsu")
class LUWChunkDecoder(Decoder):

    def decode(self, tokens: List[str], pos: List[str], labels: List[str], **kwargs) -> Dict:
        """
        labelsが以下の形式の場合に利用する
        (B or I)(B or I)(LUW品詞)_(LUW活用型)_(LUW活用形)
        """
        luw = list()
        chunk = list()
        for l in labels:
            if l in ["unk", "pad"]:
                chunk.append("B")
                luw.append("*")
            else:
                chunk.append(l[0])
                if l[1] == "B":
                    luw.append(l[2:])
                else:
                    luw.append("*")
        res = {
            "tokens": tokens,
            "pos": pos,
            "luw": luw,
            "chunk": chunk
        }
        res.update(kwargs)

        return res
    

class Encoder(Registrable):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.encode(*args, **kwds)

    def encode(self, tokens: List[str], pos: List[str], **kwargs) -> Any:
        """
        Decoderが出力する形式を受け取って、所望の出力形式に変換する
        """
        raise NotImplementedError
    

@Encoder.register("bunsetsu-split")
class BunsetsuSplitter(Encoder):

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> Any:
        c_tokens = list()
        prv = ""
        for token, c in zip(tokens, chunk):
            if c == "B":
                if len(prv) > 0:
                    c_tokens.append(prv)
                    prv = token
            else:
                prv += token

        if len(prv) > 0:
            c_tokens.append(prv)

        return " ".join(c_tokens)
    

@Encoder.register("luw-split")
class LUWSplitter(Encoder):

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> Any:
        c_tokens = list()
        prv = ""
        for token, c in zip(tokens, pos):
            if c != "*":
                if len(prv) > 0:
                    c_tokens.append(prv)
                    prv = token
            else:
                prv += token

        if len(prv) > 0:
            c_tokens.append(prv)
            
        return " ".join(c_tokens)
    

class SUWTokenizer(Registrable):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def tokenize(self, sentence: str) -> Dict:
        """
        以下のフォーマットを必ず含む結果を返す。（他にフィールドがあっても良い)
        {
            "sentence": "",
            "tokens": [],
            "pos": []
        }
        """
        raise NotImplementedError
    

@SUWTokenizer.register("mecab")
class MecabSUWTokenizer(SUWTokenizer):

    def __init__(self, 
        mecab_dic: Optional[str] = "ipadic",
        mecab_option: Optional[str] = None) -> None:
        super().__init__()

        mecab_option = mecab_option or ""
        self.mecab = fugashi.GenericTagger(mecab_option)

    def tokenize(self, sentence: str) -> Dict:
        return super().tokenize(sentence)


class Predictor:

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)

        self.model = LUWParserModel.by_name(self.config["model_name"]).from_config(self.config["model_config"], **self.config["dataeset_options"])
        self.model.load_state_dict(torch.load(self.find_best_pt(model_dir)))

        self.decoder = Decoder.by_name(self.config["model_config"]["decoder"])()

        self.dataeset_options = self.config['dataeset_options']
        self.dataeset_options["label_file"] = os.path.join(model_dir, "labels.json")

        posfile = os.path.join(model_dir, "pos.json")
        if os.path.exists(posfile):
            self.dataeset_options["pos_file"] = posfile

    @staticmethod
    def find_best_pt(model_dir: str) -> str:
        candidates = list()
        longest = -1
        idx = -1
        for i, fname in enumerate(glob.glob(os.path.join(model_dir, "best_at_*.pt"))):
            candidates.append(fname)
            base = os.path.basename(fname)
            epoch = base.split("_")[2]
            epoch = int(epoch.split(".")[0])
            if longest < epoch:
                longest = epoch
                idx = i

        return candidates[idx]
