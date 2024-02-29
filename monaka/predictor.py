# -*- coding: utf-8 -*-

import os
import glob
import json
import torch
import fugashi
import numpy as np

from typing import List, Dict, Any, Optional
from registrable import Registrable

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from monaka.model import LUWParserModel, init_device, is_master
from monaka.dataset import LUWJsonLDataset
from monaka.metric import MetricReporter

BASE_DIR = os.path.abspath(os.path.dirname(__file__)) # monaka dir
RESC_DIR = os.path.join(BASE_DIR, "resource") # monaka/resource dir

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
            "luw": luw[:len(tokens)],
            "chunk": chunk[:len(tokens)]
        }
        res.update(kwargs)

        return res
    

@Decoder.register("comainu")
class ComainuDecoder(Decoder):

    def decode(self, tokens: List[str], pos: List[str], labels: List[str], **kwargs) -> Dict:
        """
        labelsが以下の形式の場合に利用する Comainu方式
        (B or I)(B or I)(a)
        """
        luw = list()
        chunk = list()
        prv = -1
        for l, p in zip(labels, pos):
            if l in ["unk", "pad"]:
                chunk.append("B")
                luw.append(p)
            else:
                chunk.append(l[0])
                if l[1] == "B":
                    luw.append(p)
                    prv = len(luw) -1
                elif l[1:] == "Ia":
                    if prv > 0:
                        luw[prv] = p
                    luw.append("*")
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
    

@Encoder.register("jsonl")
class PassThrough(Encoder):

    def encode(self, tokens: List[str], pos: List[str], **kwargs) -> Any:
        r = {"tokens": tokens, "pos": pos}
        r.update(kwargs)
        return json.dumps(r, ensure_ascii=False)


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
        dic: Optional[str] = "gendai") -> None:
        super().__init__()

        dicdir = os.path.join(RESC_DIR, dic)
        mecabrc = os.path.join(RESC_DIR, "mecabrc")
        mecab_option = f"-r {mecabrc} -d {dicdir}"
        self.mecab = fugashi.GenericTagger(mecab_option)

    @staticmethod
    def get_pos(feature):
        res = list()
        for feat in feature[:5]:
            if "*" not in feat:
                res.append(feat)
        return "-".join(res)


    def tokenize(self, sentence: str) -> Dict:
        res = {
            "sentence": sentence,
            "tokens": [],
            "pos": [],
            "features": []
        }
        for word in self.mecab(sentence):
            res["tokens"].append(word.surface)
            res["pos"].append(self.get_pos(word.feature))
            res["features"].append(word.feature)
        return res


class Predictor:

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)

        self.model = LUWParserModel.by_name(self.config["model_name"]).from_config(self.config["model_config"], **self.config["dataeset_options"])
        self.model.load_state_dict(torch.load(self.find_best_pt(model_dir)))
        self.model.eval()

        self.decoder = Decoder.by_name(self.config["model_config"]["decoder"])()

        self.dataeset_options = self.config['dataeset_options']
        self.dataeset_options["label_file"] = os.path.join(model_dir, "labels.json")

        with open(self.dataeset_options["label_file"]) as f:
            self.label_dic = json.load(f)

        self.inv_label_dic = {v:k for k, v in self.label_dic.items()}

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
    
    def extract_labels(self, word_ids, labels):
        res = list()
        if word_ids is None:
            return [self.inv_label_dic.get(l, "unk") for l in labels]
        prv = -1
        for wid, l in zip(word_ids, labels):
            if wid is not None and wid >= 0:
                if wid == prv:
                    continue
                res.append(self.inv_label_dic.get(l, "unk"))
                wid = prv
        return res

    def predict(self, input: List[str], suw_tokenizer: str, suw_tokenizer_option: dict, encoder_name: str, batch_size: int = 8, device: str="cpu"):
        encoder = Encoder.by_name(encoder_name)()
        tokenizer = SUWTokenizer.by_name(suw_tokenizer)(**suw_tokenizer_option)

        data = [tokenizer.tokenize(sent) for sent in input]
        dataset = LUWJsonLDataset(data, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LUWJsonLDataset.collate_function)


        init_device(device)
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)


        for data in dataloader:
            #word_ids = [sbw.word_ids() for sbw in data["subwords"]]
            subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(device)
            word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
            pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None

            out = self.model(subwords, word_ids, pos_ids)
            pred = torch.argmax(out, dim=-1) # batch, len, 

            pred_np = pred.detach().cpu().numpy()
            for prd, wids, sentence, tokens, pos in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"]):
                if not dataset.label_for_all_subwords:
                    labels = self.extract_labels(None, prd)
                else:
                    labels = self.extract_labels(wids, prd)
                res = self.decoder.decode(tokens, pos, labels)
                res["sentence"] = sentence
                yield encoder.encode(**res)

    def evaluate(self, inputfile: str, batch_size: int = 8, device: str="cpu", targets: List[str]=("luw", "chunk")):

        dataset = LUWJsonLDataset(inputfile, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LUWJsonLDataset.collate_function)

        init_device(device)
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)

        reporters = {name: MetricReporter(name) for name in targets}


        for data in dataloader:
            subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(device)
            word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
            pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None

            out = self.model(subwords, word_ids, pos_ids)
            pred = torch.argmax(out, dim=-1) # batch, len, 

            pred_np = pred.detach().cpu().numpy()
            for prd, wids, sentence, tokens, pos, gold in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"], data["labels"]):
                if not dataset.label_for_all_subwords:
                    labels = self.extract_labels(None, prd)
                else:
                    labels = self.extract_labels(wids, prd)
                res = self.decoder.decode(tokens, pos, labels)
                gres = self.decoder.decode(tokens, pos, gold)

                if np.random.random() < 0.02:
                    print("gold", gres)
                    print("pred", res)
                for target in targets:
                    rep = reporters[target]
                    rep.update(gres[target], res[target])

        for rep in reporters.values():
            rep.pretty()