# -*- coding: utf-8 -*-

import os
import io
import re
import sys
import csv
import glob
import json
import torch
import ipadic
import fugashi
import numpy as np

from typing import List, Dict, Any, Optional
from registrable import Registrable

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from monaka.model import LUWParserModel, init_device, is_master
from monaka.dataset import LUWJsonLDataset, LemmaJsonDataset
from monaka.metric import MetricReporter, SpanBasedMetricReporter
from monaka.mylogging import logger

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
            pos.extend([t for t in token.split("-") if len(t) > 0])
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
        (B or I)(B or I)(LUW品詞)
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
        if len(chunk) > 0: # 銭湯は必ずB
            chunk[0] = 'B'
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

    def __init__(self, **kwargs) -> None:
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

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], features: List[List[str]], **kwargs) -> Any:
        output = io.StringIO()
        writer = csv.writer(output)
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        for token, p, l, c, f in zip(tokens, pos, lpos, chunk, features):
            writer.writerow((token, p, f[5], l, c))

        return output.getvalue()
    
@Encoder.register("mecab")
class MeCabEncoder(Encoder):
    
    def __init__(self, node_format: str='%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[13]\t%f[27]\t%f[28]\n', bos_format: str='', eos_format: str='\n', unk_format: str='%m\t%m\t%m\t%m\tUNK\t%f[4]\t%f[5]\t\n', eon_format: str='', **kwargs) -> None:
        super().__init__(**kwargs)
        self.node_format = node_format
        self.bos_format = bos_format
        self.eos_format = eos_format
        self.unk_format = unk_format
        self.eon_format = eon_format

        self.space_matcher = re.compile(r'^(\s+)')
        self.f_matcher = re.compile(r'\%f\[([\d\,]+)\]')
        self.fc_matcher = re.compile(r'\%F(.+)\[([\d\,]+)\]')

    def format(self, fstring: str, sentence: str, m_type, token: str, p, l, c, feat: List[str], start: int)-> str:
        """
        format like mecab

        %s	形態素種類 (0: 通常, 1: 未知語, 2:文頭, 3:文末)
        %S	入力文
        %L	入力文の長さ
        %m	形態素の表層文字列
        %M	形態素の表層文字列, ただし空白文字も含めて出力 (%pS を参照のこと)
        %h	素性の内部 ID
        %%	% そのもの
        %c	単語生起コスト
        %H	素性 (品詞, 活用, 読み) 等を CSV で表現したもの
        %t	文字種 id
        %P	周辺確率 (-l2 オプションを指定したときのみ有効)
        %pi	形態素に付与されるユニークなID
        %pS	もし形態素が空白文字列で始まる場合は, その空白文字列を表示 %pS%m と %M は同一
        %ps	開始位置
        %pe	終了位置
        %pC	1つ前の形態素との連接コスト
        %pw	%c と同じ
        %pc	連接コスト + 単語生起コスト (文頭から累積)
        %pn	連接コスト + 単語生起コスト (その形態素単独, %pw + %pC)
        %pb	最適パスの場合 *, それ以外は ' '
        %pP	周辺確率 (-l2 オプションを指定したときのみ有効)
        %pA	blpha, forward log 確率 (-l2 オプションを指定したときのみ有効)
        %pB	beta, backward log 確率 (-l2 オプションを指定したときのみ有効)
        %pl	形態素の表層文字列としての長さ, strlen (%m) と同一
        %pL	形態素の表層文字列としての長さ, ただし空白文字列も含む, strlen(%M) と同一
        %phl	左文脈 id
        %phr	右文脈 id
        %b  文節情報
        %l  長単位情報
        %f[N]	csv で表記された素性の N番目の要素
        %f[N1,N2,N3...]	N1,N2,N3番目の素性を, "," を デリミタとして表示
        %FC[N1,N2,N3...]	N1,N2,N3番目の素性を, C を デリミタとして表示.
        ただし, 要素が 空の場合は以降表示が省略される. (例)F-[0,1,2]
        \0 \a \b \t \n \v \f \r \\	通常の エスケープ文字列
        \s	' ' (半角スペース)
        設定ファイルに記述するときに使用
        """
        f = list()
        f.extend(feat)
        #f.extend((l, c))

        output = fstring.replace('\\0', '\0')
        output = output.replace('\\a', '\a')
        output = output.replace('\\b', '\b')
        output = output.replace('\\t', '\t')
        output = output.replace('\\n', '\n')   
        output = output.replace('\\v', '\v')
        output = output.replace('\\f', '\f')
        output = output.replace('\\r', '\r')
        output = output.replace('\\\\', '\\')

        output = output.replace('%S', sentence)
        output = output.replace('%s', str(m_type))
        output = output.replace('%L', str(len(sentence)))
        output = output.replace('%m', token.strip())
        output = output.replace('%M', token)
        output = output.replace('%h', '0')
        output = output.replace('%%', '%')
        output = output.replace('%c', '1')
        output = output.replace('%H', ','.join(f))
        output = output.replace('%t', '0')
        output = output.replace('%P', '0')
        output = output.replace('%pi', '0')
        # %pS
        m = self.space_matcher.match(token)
        if m:
            output = output.replace('%pS', m.group(1))
        else:
            output = output.replace('%pS', '')

        output = output.replace('%ps', str(start))
        output = output.replace('%pe', str(start + len(token)))
        output = output.replace('%pC', '0')
        output = output.replace('%pw', '0')
        output = output.replace('%pc', '0')
        output = output.replace('%pn', '0')
        output = output.replace('%pC', '*')
        output = output.replace('%pP', '0')
        output = output.replace('%pB', '0')
        output = output.replace('%pl', str(len(token.strip())))
        output = output.replace('%pL', str(len(token)))
        output = output.replace('%phl', '0')
        output = output.replace('%phr', '0')
        output = output.replace('%b', c)
        output = output.replace('%l', l)
        output = output.replace('\s', ' ')
        
        for m in self.f_matcher.finditer(output):
            txt = m.group(0)
            index = [int(v) for v in m.group(1).split(',')]
            vals = [f[i] if i < len(f) else '*' for i in index]
            output = output.replace(txt, ','.join(vals), 1)

        for m in self.fc_matcher.finditer(output):
            txt = m.group(0)
            sep = m.group(1)
            index = [int(v) for v in m.group(2).split(',')]
            vals = [f[i] for i in index if i < len(f) and f[i] not in ('', '*')]
            output = output.replace(txt, sep.join(vals), 1)

        return output

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], features: List[List[str]], **kwargs) -> Any:
        output = ''
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        output += self.format(self.bos_format, kwargs.get('sentence', ''), 0, '', '', '', '', [], 0)
        start = 0
        for i, (token, p, l, c, f) in enumerate(zip(tokens, pos, lpos, chunk, features)):
            m_type = 0
            if i == 0:
                m_type = 2
            elif i == len(tokens) -1:
                m_type = 3
            output += self.format(self.node_format, kwargs.get('sentence', ''), m_type, token, p, l, c, f, start)
            start += len(token)

        output += self.format(self.eos_format, kwargs.get('sentence', ''), 0, '', '', '', '', [], len(tokens))
        return output
    
    
@Encoder.register("cabocha")
class CaboChaEncoder(Encoder):

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str],  features: List[List[str]], **kwargs) -> Any:
        chunk_count = 0
        outputs = list()
        for token, c, feat in zip(tokens, chunk, features):
            if chunk_count < 1:
                outputs.append("* 0 -1D")
                chunk_count += 1
            elif 'B' in c:
                outputs.append(f"* {chunk_count} -1D")
                chunk_count += 1
            outputs.append(f"{token.strip()}\t{','.join(feat)}")
        outputs.append('EOS')
        return '\n'.join(outputs)
    

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

@Encoder.register("bccwj")
class BCCWJComainu(Encoder):
    FIELDS = [
        "file(S)",
        "start(S)",
        "end(S)",
        "boundary(S)",
        "orthToken(S)",
        "reading(S)",
        "lemma(S)",
        "meaning(S)",
        "pos(S)",
        "cType(S)",
        "cForm(S)",
        "usage(S)",
        "pronToken(S)",
        "pronBase(S)",
        "kana(S)",
        "kanaBase(S)",
        "form(S)",
        "formBase(S)",
        "formOrthBase(S)",
        "formOrth(S)",
        "orthBase(S)",
        "wType(S)",
        "charEncloserOpen(S)",
        "charEncloserClose(S)",
        "originalText(S)",
        "order",
        "BOB",
        "LUW",
        "l_orthToken",
        "l_reading",
        "l_lemma",
        "l_pos",
        "l_cType",
        "l_cForm",
        "depend",
        "MID",
        "m"
    ]

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> Any:
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        meta = kwargs.get("meta")
        outs = list()
        start = -1
        count = 0
        for p, l, c, m in zip(pos, lpos, chunk, meta):
            out = [m.get(name, "") for name in self.FIELDS]
            out[self.FIELDS.index("BOB")] = c
            if start < 0 or '*' not in l: #長単位先頭
                out[self.FIELDS.index("LUW")] = 'B'
                out[self.FIELDS.index("l_orthToken")] = m['orthToken(S)']
                out[self.FIELDS.index("l_reading")] = m['reading(S)']
                out[self.FIELDS.index("l_pos")] = l
                out[self.FIELDS.index("l_cType")] = m['cType(S)']
                out[self.FIELDS.index("l_cForm")] = m['cForm(S)']
                start = count
            else: #長単位途中
                out[self.FIELDS.index("LUW")] = 'I'
                outs[start][self.FIELDS.index("l_orthToken")] += m['orthToken(S)'] # 長単位先頭のトークンに追記
                out[self.FIELDS.index("l_orthToken")] = "*"
                outs[start][self.FIELDS.index("l_reading")] += m['reading(S)'] # 長単位先頭のトークンに追記
                out[self.FIELDS.index("l_reading")] = "*"
                out[self.FIELDS.index("l_pos")] = "*"
                outs[start][self.FIELDS.index("l_cType")] = m['cType(S)'] # 長単位先頭のトークンを上書き
                out[self.FIELDS.index("l_cType")] = "*"
                outs[start][self.FIELDS.index("l_cForm")] = m['cForm(S)'] # 長単位先頭のトークンを上書き
                out[self.FIELDS.index("l_cForm")] = "*"

            outs.append(out)
            count += 1
        return "\n".join(['\t'.join(o) for o in outs])


@Encoder.register("bcpexport")
class BCPExport(Encoder):
    FIELDS = [
        "corpusName(S)",
        "file(S)",
        "start(S)",
        "end(S)",
        "boundary(S)",
        "orthToken(S)",
        "pronToken(S)",
        "reading(S)",
        "lemma(S)",
        "originalText(S)",
        "pos(S)",
        "sysCType(S)",
        "cForm(S)",
        "apply(S)",
        "additionalInfo(S)",
        "lid(S)",
        "meaning(S)",
        "UpdUser(S)",
        "UpdDate(S)",
        "order(S)",
        "note(S)",
        "open(S)",
        "close(S)",
        "wType(S)",
        "fix(S)",
        "variable(S)",
        "formBase(S)",
        "lemmaID(S)",
        "usage(S)",
        "sentenceId(S)",
        "s_memo(S)",
        "origChar(S)",
        "pSampleID(S)",
        "pStart(S)",
        "orthBase(S)",
        "file(L)",
        "l_orthToken(L)",
        "l_pos(L)",
        "l_cType(L)",
        "l_cForm(L)",
        "l_reading(L)",
        "l_lemma(L)",
        "luw(L)",
        "memo(L)",
        "UpdUser(L)",
        "UpdDate(L)",
        "l_start(L)",
        "l_end(L)",
        "bunsetsu1(L)",
        "bunsetsu2(L)",
        "corpusName(L)",
        "diffSuw(L)",
        "l_lemmaNew(L)",
        "l_readingNew(L)",
        "l_orthBase(L)",
        "l_formBase(L)",
        "l_pronToken(L)",
        "l_wType(L)",
        "l_originalText(L)",
        "complex(L)",
        "l_meaning(L)",
        "l_kanaToken(L)",
        "l_formOrthBase(L)",
        "l_origChar(L)",
        "note(L)",
        "pSampleID(L)",
        "pStart(L)",
        "rn"
    ]

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> Any:
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        meta = kwargs.get("meta")
        outs = list()
        start = -1
        count = 0
        for p, l, c, m in zip(pos, lpos, chunk, meta):
            out = [m.get(name, "") for name in self.FIELDS]
            out[self.FIELDS.index("bunsetsu1(L)")] = c
            if start < 0 or '*' not in l: #長単位先頭
                out[self.FIELDS.index("luw(L)")] = 'B'
                out[self.FIELDS.index("l_orthToken(L)")] = m['orthToken(S)']
                out[self.FIELDS.index("l_reading(L)")] = m['reading(S)']
                out[self.FIELDS.index("l_pos(L)")] = l
                out[self.FIELDS.index("l_cType(L)")] = m['sysCType(S)']
                out[self.FIELDS.index("l_cForm(L)")] = m['cForm(S)']
                start = count
            else: #長単位途中
                out[self.FIELDS.index("luw(L)")] = 'I'
                outs[start][self.FIELDS.index("l_orthToken(L)")] += m['orthToken(S)'] # 長単位先頭のトークンに追記
                out[self.FIELDS.index("l_orthToken(L)")] = "*"
                outs[start][self.FIELDS.index("l_reading(L)")] += m['reading(S)'] # 長単位先頭のトークンに追記
                out[self.FIELDS.index("l_reading(L)")] = "*"
                out[self.FIELDS.index("l_pos(L)")] = "*"
                outs[start][self.FIELDS.index("l_cType(L)")] = m['sysCType(S)'] # 長単位先頭のトークンを上書き
                out[self.FIELDS.index("l_cType(L)")] = "*"
                outs[start][self.FIELDS.index("l_cForm(L)")] = m['cForm(S)'] # 長単位先頭のトークンを上書き
                out[self.FIELDS.index("l_cForm(L)")] = "*"

            outs.append(out)
            count += 1
        return "\n".join(['\t'.join(o) for o in outs])

    

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
        if dic == 'ipadic':
            self.mecab = fugashi.GenericTagger(ipadic.MECAB_ARGS)
        else:
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

    def predict(self, input: List[str], suw_tokenizer: str, suw_tokenizer_option: dict, encoder_name: str, batch_size: int = 8, device: str="cpu", **kwargs):
        encoder = Encoder.by_name(encoder_name)(**kwargs)
        tokenizer = SUWTokenizer.by_name(suw_tokenizer)(**suw_tokenizer_option)

        data = [tokenizer.tokenize(sent) for sent in input]

        self.dataeset_options['store_all'] = True
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
            for prd, wids, sentence, tokens, pos, feat, skip in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"], data["features"], data["skip"]):
                if not skip:
                    if not dataset.label_for_all_subwords:
                        labels = self.extract_labels(None, prd)
                    else:
                        labels = self.extract_labels(wids, prd)
                    res = self.decoder.decode(tokens, pos, labels)
                else:
                    res = self.decoder.decode(tokens, pos, ['BB*' for _ in tokens])
                res["sentence"] = sentence
                res["features"] = feat
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


class EnsemblePredictor:

    def __init__(self, model_dirs: List[str], device: str="cpu", mask: Optional[str]=None) -> None:
        self.model_dirs = model_dirs
        
        with open(os.path.join(model_dirs[0], "config.json")) as f:
            self.config = json.load(f)

        self.config["dataeset_options"]["label_file"] = os.path.join(model_dirs[0], "labels.json")

        posfile = os.path.join(model_dirs[0], "pos.json")
        if os.path.exists(posfile):
            self.config["dataeset_options"]["pos_file"] = posfile

        self.models = list()
        for model_dir in model_dirs:
            model = LUWParserModel.by_name(self.config["model_name"]).from_config(self.config["model_config"], **self.config["dataeset_options"])
            model.load_state_dict(torch.load(self.find_best_pt(model_dir)), strict=False)
            model.eval()
            self.models.append(model)

        self.decoder = Decoder.by_name(self.config["model_config"]["decoder"])()

        self.dataeset_options = self.config['dataeset_options']
        self.dataeset_options["fold_sentence"] = True

        with open(self.dataeset_options["label_file"]) as f:
            self.label_dic = json.load(f)

        self.inv_label_dic = {v:k for k, v in self.label_dic.items()}

        init_device(device)
        try:
            device = int(device)
        except:
            pass
        for model in self.models:
            model.to(device)
        self.device = device

        if mask is not None:
            try:
                with open(mask) as f:
                    self.mask = json.load(f)
                with open(posfile) as f:
                    self.pos_dic = json.load(f)
                    self.inv_pos = {v:k for k, v in self.pos_dic.items()}
            except:
                pass


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

    def predict(self, input: List[str], suw_tokenizer: str, suw_tokenizer_option: dict, encoder_name: str, batch_size: int = 8, **kwargs):
        encoder = Encoder.by_name(encoder_name)(**kwargs)
        tokenizer = SUWTokenizer.by_name(suw_tokenizer)(**suw_tokenizer_option)

        data = [tokenizer.tokenize(sent) for sent in input]

        self.dataeset_options['store_all'] = True
        dataset = LUWJsonLDataset(data, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LUWJsonLDataset.collate_function)


        for data in dataloader:
            #word_ids = [sbw.word_ids() for sbw in data["subwords"]]
            subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(self.device)
            word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(self.device)
            pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(self.device) if "pos_ids" in data else None
            lemma_ids = pad_sequence(data["lemma_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(self.device) if "lemma_ids" in data else None
            lemma_word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["lemma_subwords"]], batch_first=True, padding_value=-1).to(self.device) if "lemma_ids" in data else None

            # average ensemble
            out = 0.
            for model in self.models:
                out = out + model(subwords, word_ids, pos_ids)
            pred = torch.argmax(out, dim=-1) # batch, len, 

            pred_np = pred.detach().cpu().numpy()
            for prd, wids, sentence, tokens, pos, feat, skip in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"], data["features"], data["skip"]):
                if not skip:
                    if not dataset.label_for_all_subwords:
                        labels = self.extract_labels(None, prd)
                    else:
                        labels = self.extract_labels(wids, prd)
                    res = self.decoder.decode(tokens, pos, labels)
                else:
                    res = self.decoder.decode(tokens, pos, ['BB*' for _ in tokens])
                res["sentence"] = sentence
                res["features"] = feat
                yield encoder.encode(**res)

    def apply_single_suw_rule(self, decoder_out, top):
        singles = list(range(len(decoder_out["luw"])))
        for i, l in enumerate(decoder_out["luw"]):
            if '*' in l:
                singles[i] = -1
                if i > 0:
                    singles[i-1] = -1

        for s, pos, luw, t in zip(singles, decoder_out['pos'], decoder_out['luw'], top):
            if s < 0:
                continue
            if '可能' not in pos:
                decoder_out['luw'][s] = pos
            elif '名詞-普通名詞-助数詞可能' in pos:
                decoder_out['luw'][s] = '名詞-普通名詞-一般'
            elif '動詞-非自立可能' in pos:
                decoder_out['luw'][s] = '動詞-一般'
            elif '形容詞-非自立可能' in pos:
                decoder_out['luw'][s] = '形容詞-一般'
            elif '名詞-普通名詞-サ変可能' in pos:
                decoder_out['luw'][s] = '名詞-普通名詞-一般'
            elif '名詞-普通名詞-形状詞可能' in pos or '名詞-普通名詞-サ変形状詞可能' in pos:
                for i in t:
                    label = self.inv_label_dic[i]
                    if '形状詞-一般' in label:
                        decoder_out['luw'][s] = '形状詞-一般'
                        break
                    elif '名詞-普通名詞-一般' in label:
                        decoder_out['luw'][s] = '名詞-普通名詞-一般'
                        break
            elif '名詞-普通名詞-副詞可能' in pos :
                for i in t:
                    label = self.inv_label_dic[i]
                    if '副詞' in label:
                        decoder_out['luw'][s] = '副詞'
                        break
                    elif '名詞-普通名詞-一般' in label:
                        decoder_out['luw'][s] = '名詞-普通名詞-一般'
                        break
        return decoder_out
        

    def predict_raw(self, input, encoder_name: str, batch_size: int = 8):
        encoder = Encoder.by_name(encoder_name)()

        dataset = LUWJsonLDataset(input, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=LUWJsonLDataset.collate_function)


        for data in dataloader:
            #word_ids = [sbw.word_ids() for sbw in data["subwords"]]
            subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(self.device)
            word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(self.device)
            pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(self.device) if "pos_ids" in data else None

            # average ensemble
            out = 0.
            for model in self.models:
                out = out + model(subwords, word_ids, pos_ids)

            pred = torch.argmax(out, dim=-1) # batch, len, 
            tops = torch.topk(out, len(self.label_dic), dim=-1)

            pred_np = pred.detach().cpu().numpy()
            tops_np = tops.indices.detach().cpu().numpy()
            prv_tokens = None
            prv_pos = None
            prv_labels = None
            for prd, wids, sentence, tokens, pos, meta, fold, top in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"], data.get("meta", data["pos"]), data["fold"], tops_np):
                if not dataset.label_for_all_subwords:
                    labels = self.extract_labels(None, prd)
                else:
                    labels = self.extract_labels(wids, prd)
                if fold < 0:
                    res = self.decoder.decode(tokens, pos, labels)
                    res = self.apply_single_suw_rule(res, top)
                    res["sentence"] = sentence
                    res["meta"] = meta
                    out = encoder.encode(**res)
                    yield out
                elif fold == 0:
                    logger.warning(f"fold: 0 {''.join(tokens)}")
                    prv_tokens = tokens
                    prv_pos = pos
                    prv_labels = labels
                else: #fold == 1
                    logger.warning(f"fold: 1 {''.join(tokens)}")
                    prv_tokens.extend(tokens)
                    prv_pos.extend(pos)
                    prv_labels.extend(labels)
                    logger.warning(f"unfolding {''.join(prv_tokens)}")
                    res = self.decoder.decode(prv_tokens, prv_pos, prv_labels)
                    res = self.apply_single_suw_rule(res, top)
                    res["sentence"] = sentence
                    res["meta"] = meta
                    out = encoder.encode(**res)
                    yield out


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

    def predict(self, input: Dict, use_pos: bool=False) -> str:
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
        #print(inp)
        if '<unk>' in inp:
            return ''.join([v['lemma'] for v in data.get('suw', [{'lemma': ''}])])
        
        if use_pos:
            pos:str = data.get('pos', '')
            if pos.startswith(('接続詞', '感動詞', '副詞')):
                # 特定の品詞の場合はSUWを直接使う
                return ''.join([v['lemma'] for v in data.get('suw', [{'lemma': ''}])])
            
        outputs = self.model.generate(data['input_ids'].unsqueeze(0).to(self.device), 
                                      attention_mask=data['attention_mask'].to(self.device),
                                      do_sample=False)
        #print(outputs)
        #print(self.tokenizer.decode(outputs[0], skip_special_tokens=False))
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

