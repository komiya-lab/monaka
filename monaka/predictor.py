# -*- coding: utf-8 -*-

import os
import io
import sys
import csv
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
from monaka.dataset import LUWJsonLDataset, LemmaJsonDataset
from monaka.metric import MetricReporter, SpanBasedMetricReporter

BASE_DIR = os.path.abspath(os.path.dirname(__file__)) # monaka dir
RESC_DIR = os.path.join(BASE_DIR, "resource") # monaka/resource dir

class Decoder(Registrable):

    def __init__(self) -> None:
        super().__init__()


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.decode(*args, **kwds)

    def luw_pos(self, text: str, pos_level: int) -> str:
        pos: List = list()
        for token in text.split("_"):
            pos.extend([t for t in token.split("-") if len(t) > 1])
        if pos_level is not None and pos_level > -1:
            return "-".join(pos[:pos_level])
        return "-".join(pos)
    
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

    def decode(self, tokens: List[str], pos: List[str], labels: List[str], pos_level:int = -1, **kwargs) -> Dict:
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
                    luw.append(self.luw_pos(l[2:], pos_level))
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

    def decode(self, tokens: List[str], pos: List[str], labels: List[str], pos_level:int = -1, **kwargs) -> Dict:
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
                luw.append(self.luw_pos(p, pos_level))
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
    

def append_spans(data):
    start = 0
    data["suw_span"] = []
    for token in data["tokens"]:
        data["suw_span"].append((start, start+len(token)))
        start += len(token)

    data["luw_span"] = []
    data["luw_triples"] = []
    data["chunk_span"] = []

    luw_start = -1
    luw_end = -1
    luw_type = None
    chunk_start = -1
    chunk_end = -1
    for span, luw, chunk in zip(data["suw_span"], data["luw"], data["chunk"]):
        if luw_start < 0:
            luw_start = span[0]
            luw_end = span[1]
            chunk_start = span[0]
            chunk_end = span[1]
            luw_type = luw
            continue
        if "*" in luw:
            luw_end = span[1]
        else:
            data["luw_span"].append((luw_start, luw_end))
            data["luw_triples"].append((luw_start, luw_end, luw_type))
            luw_start = span[0]
            luw_end = span[1]
            luw_type = luw
        
        if "I" in chunk:
            chunk_end = span[1]
        else:
            data["chunk_span"].append((chunk_start, chunk_end))
            chunk_start = span[0]
            chunk_end = span[1]

    data["chunk_span"].append((chunk_start, chunk_end))
    data["luw_span"].append((luw_start, luw_end))
    data["luw_triples"].append((luw_start, luw_end, luw_type))

    return data
    
@Encoder.register("jsonl")
class PassThrough(Encoder):

    def encode(self, tokens: List[str], pos: List[str], **kwargs) -> Any:
        r = {"tokens": tokens, "pos": pos}
        r.update(kwargs)
        append_spans(r)
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
    

@Encoder.register("csv")
class CSVEncoder(Encoder):

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> Any:
        output = io.StringIO()
        writer = csv.writer(output)
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        for tpl in zip(tokens, pos, lpos, chunk):
            writer.writerow(tpl)

        return output.getvalue()
    

@Encoder.register("luw-split")
class LUWSplitter(Encoder):

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> Any:
        c_tokens = list()
        prv = ""
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        for token, c in zip(tokens, lpos):
            if c != "*":
                if len(prv) > 0:
                    c_tokens.append(prv)
                prv = token

            else:
                prv += token

        if len(prv) > 0:
            c_tokens.append(prv)
            
        return " ".join(c_tokens)
    
@Encoder.register("mrp")
class MRPformatter(Encoder):

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> Any:
        pass


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

        self.config["dataeset_options"]["label_file"] = os.path.join(model_dir, "labels.json")

        posfile = os.path.join(model_dir, "pos.json")
        if os.path.exists(posfile):
            self.config["dataeset_options"]["pos_file"] = posfile

        self.model = LUWParserModel.by_name(self.config["model_name"]).from_config(self.config["model_config"], **self.config["dataeset_options"])
        self.model.load_state_dict(torch.load(self.find_best_pt(model_dir)), strict=False)
        self.model.eval()

        self.decoder = Decoder.by_name(self.config["model_config"]["decoder"])()

        self.dataeset_options = self.config['dataeset_options']

        with open(self.dataeset_options["label_file"]) as f:
            self.label_dic = json.load(f)

        self.inv_label_dic = {v:k for k, v in self.label_dic.items()}


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
                prv = wid
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
            lemma_ids = pad_sequence(data["lemma_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device) if "lemma_ids" in data else None
            lemma_word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["lemma_subwords"]], batch_first=True, padding_value=-1).to(device) if "lemma_ids" in data else None

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

    def predict_raw(self, input: str, encoder_name: str, batch_size: int = 8, device: str="cpu"):
        print(input, file=sys.stderr)
        encoder = Encoder.by_name(encoder_name)()

        dataset = LUWJsonLDataset(input, **self.dataeset_options)
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

    def evaluate(self, inputfile: str, batch_size: int = 8, device: str="cpu", targets: List[str]=("luw", "chunk"), pos_level: int = -1, format_: str="pretty", 
                suw_tokenizer: str=None, suw_tokenizer_option: dict=None, outputfile: str=None):
        tokenizer = SUWTokenizer.by_name(suw_tokenizer)(**suw_tokenizer_option) if suw_tokenizer else None
        dataset = LUWJsonLDataset(inputfile, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LUWJsonLDataset.collate_function)
        if outputfile is not None:
            output = open(outputfile, "w")

        init_device(device)
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)

        reporters = {name: MetricReporter(name) for name in targets}
        span_reporters = {f"{name}_span": SpanBasedMetricReporter(f"{name}_span") for name in targets}
        span_reporters["luw_triples"] = SpanBasedMetricReporter("luw_triples")


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
                
                res = self.decoder.decode(tokens, pos, labels, pos_level)
                res = append_spans(res)
                gres = self.decoder.decode(tokens, pos, gold, pos_level)
                gres = append_spans(gres)

                if outputfile is not None:
                    print(json.dumps(res, ensure_ascii=False), file=output)

                if np.random.random() < 0.02:
                    print("gold", gres, file=sys.stderr)
                    print("pred", res, file=sys.stderr)
                for target in targets:
                    rep = reporters[target]
                    rep.update(gres[target], res[target])

                    starget = f"{target}_span"
                    rep = span_reporters[starget]
                    rep.update(gres[starget], res[starget])
                span_reporters["luw_triples"].update(gres["luw_triples"], res["luw_triples"])

        if "pretty" in format_:
            for rep in reporters.values():
                rep.pretty()
            
            for rep in span_reporters.values():
                rep.pretty()
        else:
            res = dict()
            for k, rep in reporters.items():
                res[k] = rep.to_json()

            for k, rep in span_reporters.items():
                res[k] = rep.to_json()

            print(json.dumps(res, indent=True))

        if outputfile is not None:
            output.close()

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader

class LemmaPredictor:

    def __init__(self, model_dir: str, device='cpu') -> None:
        self.model_dir = model_dir
        self.device = device
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_dir, "last-checkpoint")).to(device)
        #print(type(self.model))
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.special_tokens = list(self.tokenizer.all_special_tokens)
        self.special_tokens.extend([" ", "▁", "_"])
        self.trainer = Seq2SeqTrainer(self.model, Seq2SeqTrainingArguments(".", per_device_eval_batch_size=8, predict_with_generate=True))

    def predict(self, input: Dict) -> str:
        dataset = LemmaJsonDataset(input, **self.config['dataset_options'])
        #print(dataset[0])
        #print(dataset[0]['input_ids'].size())
        #preds = self.trainer.predict(test_dataset=dataset)
        #preds_ = [np.argmax(prd, axis=-1) for prd in preds]
        #decoded_preds = self.tokenizer.batch_decode(preds_[0], skip_special_tokens=True)
        #return decoded_preds[0]

        #print(input)
        data = dataset[0]
        #print(data['input_ids'].size())
        inp = self.tokenizer.decode(data['input_ids'])
        print(inp)
        if '<unk>' in inp:
            return ''.join([v['lemma'] for v in data['suw']])
        outputs = self.model.generate(data['input_ids'].unsqueeze(0).to(self.device), 
                                      attention_mask=data['attention_mask'].to(self.device),
                                      do_sample=False)
        #print(outputs)
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=False))
        tokens = self.tokenizer.convert_ids_to_tokens(outputs[0], skip_special_tokens=False)
        output = []
        for t in tokens:
            if t not in self.special_tokens:
                output.append(t)
            else:
                if len(output) == 0:
                    continue
                else:
                    break
        o ="".join(output)
        #print(o.replace("▁", ""))
        return o.replace("▁", "")
    
    def tokens2output(self, tokens: List[str]) -> str:
        output = []
        for t in tokens:
            if t not in self.special_tokens:
                output.append(t)
            else:
                if len(output) == 0:
                    continue
                else:
                    break
        o ="".join(output)
        #print(o.replace("▁", ""))
        return o.replace("▁", "")
    
    def batch_predict(self, inputs: Dict) -> List[str]:
        dataset = LemmaJsonDataset(input, **self.config['dataset_options'])
        loader = DataLoader(dataset=dataset, batch_size=self.config['batch_size'], shuffle=False)
        #print(data['input_ids'].size())
        res = list()
        for data in loader:
            #print(self.tokenizer.decode(data['input_ids']))
            outputs = self.model.generate(data['input_ids'].to(self.device), 
                                        attention_mask=data['attention_mask'].to(self.device),
                                        do_sample=False)
            #print(outputs)
            #print(self.tokenizer.decode(outputs[0], skip_special_tokens=False))
            for output in outputs:
                res.append(self.tokens2output(self.tokenizer.convert_ids_to_tokens(output, skip_special_tokens=False)))
    
    def evaluate(self, jsonfile: str) -> Dict:
        dataset = LemmaJsonDataset(jsonfile, **self.config['dataset_options'])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        c = 0
        a = 0
        diff = list()
        for feat in loader:
            a += 1
            pred = self.predict(feat['input'][0])
            label = feat['target'][0]
            if pred == label:
                c += 1
            else:
                diff.append(f"input: {feat['input']}, correct: {label}, pred: {pred}")
        return {
            "count": a,
            "acc": c/a,
            "diff": diff
        }

