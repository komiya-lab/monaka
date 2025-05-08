# -*- coding: utf-8 -*-

import torch
import json
import torch.nn as nn
from typing import Dict
from registrable import Registrable
from transformers import BertTokenizerFast, AutoTokenizer


class Tokenizer(Registrable):

    def __init__(self, lm_tokenizer, add_special_tokens=False) -> None:
        self.lm_tokenizer = lm_tokenizer
        self.add_special_tokens = add_special_tokens
        self.pad_token_id = lm_tokenizer.pad_token_id
        super().__init__()

    def tokenize(self, words, max_length=None):
        tokens = self.lm_tokenizer(words, is_split_into_words=True, add_special_tokens=self.add_special_tokens, max_length=max_length, truncation=True)
        return tokens
    

@Tokenizer.register("auto")
class AutoLMTokenizer(Tokenizer):
    
    def __init__(self, name: str, add_special_tokens=False, **kwargs) -> None:
        lm_tokenizer = AutoTokenizer.from_pretrained(name)
        lm_tokenizer.clean_up_tokenization_spaces = False
        super().__init__(lm_tokenizer, add_special_tokens)


try:

    from transformers.models.bert_japanese.tokenization_bert_japanese import MecabTokenizer
    class MecabPreTokenizer(MecabTokenizer):

        def mecab_split(self,i,normalized_string):
            t=str(normalized_string)
            e=0
            z=[]
            for c in self.tokenize(t):
                s=t.find(c,e)
                e=e if s<0 else s+len(c)
                z.append((0,0) if s<0 else (s,e))
            out = [normalized_string[s:e] for s,e in z if e>0]
            if len(out) > 0:
                return out
            else:
                return [normalized_string]
        
        def pre_tokenize(self,pretok):
            pretok.split(self.mecab_split)


    class BertMecabTokenizerFast(BertTokenizerFast):

        def __init__(self, vocab_file, do_lower_case=False, tokenize_chinese_chars=False, clean_up_tokenization_spaces=False, **kwargs):
            from tokenizers.pre_tokenizers import PreTokenizer,BertPreTokenizer,Sequence
            super().__init__(vocab_file=vocab_file,do_lower_case=do_lower_case,tokenize_chinese_chars=tokenize_chinese_chars,clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)
            d=kwargs["mecab_kwargs"] if "mecab_kwargs" in kwargs else {"mecab_dic":"ipadic"}
            self._tokenizer.pre_tokenizer=Sequence([PreTokenizer.custom(MecabPreTokenizer(**d)),BertPreTokenizer()])


    @Tokenizer.register("bert-tohoku-ja")
    class BertMecabLMTokenizer(Tokenizer):

        def __init__(self, **kwargs) -> None:
            lm_tokenizer = BertMecabTokenizerFast.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
            super().__init__(lm_tokenizer)

except:
    pass


class wdict(dict):

    def __init__(self, word_id_key:str, *args):
        super().__init__(*args)
        self.word_id_key = word_id_key

    def word_ids(self):
        return self.get(self.word_id_key)

    

@Tokenizer.register("word")
class WordTokenizer(Tokenizer):

    def __init__(self, pad_token, unk_token, token_dict, add_special_tokens=False):
        with open(token_dict) as f:
            self.token_dict = json.load(f)

        self.unk_token = unk_token

        class PsuedoTokenizer:

            def __init__(self, pad_token_id):
                self.pad_token_id = pad_token_id

        super().__init__(PsuedoTokenizer(pad_token), add_special_tokens)

    def tokenize(self, words, max_length=None):
        unk = self.token_dict[self.unk_token]
        r = {
            "input_ids": [self.token_dict.get(w, unk) for w in words],
            "wids": [i for i in range(len(words))]
        }
        return wdict("wids", r)