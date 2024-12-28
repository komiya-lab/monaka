import os
import csv
import sys
import typer
import json
import enum
import numpy as np

from typing import List


app = typer.Typer()

def load_chj(fname: str, luw: bool):
    if luw:
        head = [
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
    else:
        head = [
        "file(S)",
        "start(S)",
        "end(S)",
        "order(S)",
        "boundary(S)",
        "orthToken(S)",
        "reading(S)",
        "lemma(S)",
        "meaning(S)",
        "pos(S)",
        "sysCType(S)",
        "cForm(S)",
        "usage(S)",
        "pronToken(S)",
        "pronBase(S)",
        "kanaToken(S)",
        "kanaBase(S)",
        "formToken(S)",
        "formBase(S)",
        "formOrthBase(S)",
        "formOrth(S)",
        "orthBase(S)",
        "wType(S)",
        "tagStart(S)",
        "tagEnd(S)",
        "originalText(S)",
        "iType(S)",
        "iForm(S)",
        "iConType(S)",
        "fType(S)",
        "fForm(S)",
        "fConType(S)",
        "aType(S)",
        "aConType(S)",
        "aModType(S)",
        "type(S)",
        "lid(S)",
        "lemmaID(S)",
        "bunrui(S)",
        "rn"
        ]
    # generatorにすることでメモリを節約
    with open(fname) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            yield dict(zip(head, row))

def to_sentences(data, luw:bool):
    if luw:
        targets = [
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
        "rn"]
        meta = [
            "corpusName(S)",
            "file(S)"]
    else:
        targets = [
        "start(S)",
        "end(S)",
        "order(S)",
        "boundary(S)",
        "orthToken(S)",
        "reading(S)",
        "lemma(S)",
        "meaning(S)",
        "pos(S)",
        "sysCType(S)",
        "cForm(S)",
        "usage(S)",
        "pronToken(S)",
        "pronBase(S)",
        "kanaToken(S)",
        "kanaBase(S)",
        "formToken(S)",
        "formBase(S)",
        "formOrthBase(S)",
        "formOrth(S)",
        "orthBase(S)",
        "wType(S)",
        "tagStart(S)",
        "tagEnd(S)",
        "originalText(S)",
        "iType(S)",
        "iForm(S)",
        "iConType(S)",
        "fType(S)",
        "fForm(S)",
        "fConType(S)",
        "aType(S)",
        "aConType(S)",
        "aModType(S)",
        "type(S)",
        "lid(S)",
        "lemmaID(S)",
        "bunrui(S)",
        "rn"
        ]
        meta = [
            "file(S)"]
    buf_n = list()
    buf_t = list()
    for d in data:
        n_ = d["orthToken(S)"]
            
        if d["boundary(S)"] == "B" and len(buf_n) > 0:
            dd = {"sentence": "".join(buf_n),  "tokens": buf_t}
            dd.update({k: d[k] for k in meta})
            yield dd
            buf_n.clear()
            buf_t.clear()
        
        buf_n.append(n_)
        buf_t.append({k: d[k] for k in targets})
        
    if len(buf_n) > 0:
        dd = {"sentence": "".join(buf_n), "tokens": buf_t}
        dd.update({k: d[k] for k in meta})
        yield dd
            

@app.command()
def chj2jsonl(fname: str, luw: bool=True):
    for d in to_sentences(load_chj(fname, luw), luw):
        print(json.dumps(d, ensure_ascii=False))


@app.command()
def chjstats(jsonlname: str, key: str="サブコーパス名"):
    res = dict()
    with open(jsonlname) as f:
        for line in f:
            js = json.loads(line)
            for token in js["tokens"]:
                v = token[key]
                res[v] = res.get(v, 0) + 1

    for k, v in res.items():
        print(k, v)

def chjpos(data: dict):
    return f'{data.get("pos(S)", "")}'

def chjlpos(data: dict):
    return f'{data.get("l_pos(L)", "")}'

@app.command()
def chjjsonl2luwjson(jsonlfile: str, luw: bool=True, chunk: bool=True, tokenKey: str="originalText(S)"):
    with open(jsonlfile) as f:
        for line in f:
            js = json.loads(line)
            lemma_id = 0
            res = {
                "sentence": js["sentence"],
                "tokens": [t[tokenKey] for t in js["tokens"]],
                "pos": [chjpos(t) for t in js["tokens"]],
                "lemma": [],
                "lemma_id": [],
                "labels": []
            }
            for token in js["tokens"]:
                tag = ""
                if chunk:
                    tag += "B" if token["bunsetsu1(L)"] == "B" else "I"
                if luw:
                    tag += "B" if token["luw(L)"] == "B" else "I"
                    tag += chjlpos(token)
                    res['labels'].append(tag)
                    if token["luw(L)"] == "B":
                        res["lemma_id"].append(lemma_id)
                        res["lemma"].append(token["l_lemma(L)"])
                        lemma_id += 1
                    else:
                        res["lemma_id"].append(lemma_id)

            print(json.dumps({k:v for k, v in res.items() if len(v) > 0 }, ensure_ascii=False))

@app.command()
def chjjsonl2comainu(jsonlfile: str, luw: bool=True, chunk: bool=True):
    with open(jsonlfile) as f:
        for line in f:
            js = json.loads(line)
            res = {
                "sentence": js["sentence"],
                "tokens": [t["originalText(S)"] for t in js["tokens"]],
                "pos": [chjpos(t) for t in js["tokens"]],
                "labels": []
            }
            for token in js["tokens"]:
                tag = ""
                if chunk:
                    tag += "B" if token["bunsetsu1(L)"] == "B" else "I"
                if luw:
                    tag += "B" if token["luw(L)"] == "B" else "I"
                    tag += "a" if chjlpos(token) == chjpos(token) else ""
                res['labels'].append(tag)
            print(json.dumps(res, ensure_ascii=False))

@app.command()
def jsonl2chjjsonl(predfile: str, originalfile: str):
    with open(predfile) as f:
        preds = [json.loads(line) for line in f]
    pred_dic = {d["sentence"]: d for d in preds}

    with open(originalfile) as f:
        for line in f:
            js = json.loads(line)
            if js['sentence'] in pred_dic:
                pred = pred_dic[js['sentence']]
                for i, token in enumerate(js["tokens"]):
                    luw = pred['luw'][i] if i < len(pred['luw']) else ''
                    bunsetsu = pred['chunk'][i] if i < len(pred['chunk']) else 'B'
                    token.update({
                        "bunsetsu1(L)": bunsetsu,
                        "luw(L)": "B" if luw != '*' else 'I',
                        "l_pos(L)": luw
                    })
                
            else:
                for token in js["tokens"]:
                    token.update({
                        "bunsetsu1(L)": "B",
                        "luw(L)": "B",
                        "l_pos(L)": ""
                    })
            print(json.dumps(js, ensure_ascii=False))


@app.command()
def merge_jsonl(basefile: str, updatefile: str, suffix: str=''):
    keys = None

    with open(basefile) as fb, open(updatefile) as fu:
        for lb, lu in zip(fb, fu):
            jsb = json.loads(lb)
            jsu = json.loads(lu)
            for tb, tu in zip(jsb['tokens'], jsu['tokens']):
                if keys is None:
                    keys = list(tb.keys())
                udic = {f"{k}_{suffix}": v for k, v in tu.items() if k not in keys}
                tb.update(udic)
            print(json.dumps(jsb, ensure_ascii=False))

class SampleStrategy(str, enum.Enum):
    random = 'random'
    different = 'different'
    same = 'same'

@app.command()
def sample(jsonlfile: str, num: int=100, strategy: SampleStrategy=SampleStrategy.random, token_th: int= 5):
    res = list()
    with open(jsonlfile) as f:
        for line in f:
            js = json.loads(line)
            if len(js["tokens"]) < token_th:
                continue
            if strategy == SampleStrategy.different or strategy == SampleStrategy.same:
                vals = list()
                for token in js['tokens']:
                    count = [v for k, v in token.items() if k.startswith("bunsetsu")]
                    b_count = len([v for v in count if v == 'B'])
                    v = np.max((b_count, len(count)-b_count)) / len(count)
                    vals.append(v)
                res.append((np.mean(v), js))
            else:
                res.append(js)

    if strategy == SampleStrategy.different:
        res.sort(key=lambda v: v[0])
        for v, js in res[:num]:
            print(v, file=sys.stderr)
            print(json.dumps(js, ensure_ascii=False))
    elif strategy == SampleStrategy.same:
        res.sort(key=lambda v: v[0], reverse=True)
        for v, js in res[:num]:
            print(v, file=sys.stderr)
            print(json.dumps(js, ensure_ascii=False))
    else:
        np.random.shuffle(res)
        for js in res[:num]:
            print(json.dumps(js, ensure_ascii=False))


@app.command()
def jsonl2chj(jsonlfile: str, sep: str="\t"):
    head = None
    writer = csv.writer(sys.stdout, delimiter=sep)
    with open(jsonlfile) as f:
        for line in f:
            js = json.loads(line)
            meta = {k:v for k, v in js.items() if k not in ("sentence", 'tokens')}
            for token in js["tokens"]:
                if head is None:
                    head = list(meta.keys()) + list(token.keys())
                    writer.writerow(head)
                token.update(meta)
                writer.writerow([token[k] for k in head])

def comainu_js(js: dict, luw: str, luw_pos: str):
    files = js['file(S)']
    p = 0
    for i, token in enumerate(js['tokens']):
        token['file(S)'] = files
        L = token[luw]
        if L.startswith('B'):
            p = i
        if len(token['cForm(S)']) > 1:
            js['tokens'][p]['l_cForm'] = token['cForm(S)']
            js['tokens'][p]['l_cType'] = token['sysCType(S)']

        js['tokens'][p]['l_orthToken'] = js['tokens'][p].get('l_orthToken', "") + token['orthToken(S)']
        js['tokens'][p]['l_reading'] = js['tokens'][p].get('l_reading', "") + token['reading(S)']
        js['tokens'][p]['l_lemma'] = js['tokens'][p].get('l_lemma', "") + token['lemma(S)']

    return js



@app.command()
def jsonl2comainu(jsonfile: str, bunsetsu: str='bunsetsu1(L)_formOrth_all_period', luw: str='luw(L)_formOrth_all_period', luw_pos: str='l_pos(L)_formOrth_all_period'):
    #file	start	end	BOS	orthToken	reading	lemma	meaning	pos	cType	cForm	usage	pronToken	pronBase	
    # kana	kanaBase	form	formBase	formOrthBase	formOrth	orthBase	wType	charEncloserOpen	charEncloserClose	
    # originalText	order	
    # BOB	LUW	l_orthToken	l_reading	l_lemma	l_pos	l_cType	l_cForm
    writer = csv.writer(sys.stdout, delimiter='\t')
    head = [
        'file(S)', 'start(S)', 'end(S)', 'boundary(S)', 'orthToken(S)', 'reading(S)', 'lemma(S)', 'meaning(S)','pos(S)', 'sysCType(S)', 'cForm(S)', 'usage(S)', 'pronToken(S)', 'pronBase(S)',
        'kanaToken(S)', 'kanaBase(S)', 'formToken(S)', 'formBase(S)', 'formOrthBase(S)', 'formOrth(S)', 'orthBase(S)', 'wType(S)', 'charEncloserOpen', 'charEncloserClose',
        'originalText(S)', 'order(S)'
    ]
    #品詞	活用型	活用形	語彙素読み	語彙素	書辞形
    with open(jsonfile) as f:
        for line in f:
            js = json.loads(line)
            js = comainu_js(js, luw, luw_pos)
            for token in js['tokens']:
                row = [token.get(h, "") for h in head]
                L = token[luw]
                Lpos = token[luw_pos]
                pos = token['pos(S)']
                lemma = row[6]
                if '-' in lemma:
                    print(lemma, file=sys.stderr)
                    tk = lemma.split('-')
                    lemma = tk[0]
                    meaning = '-'.join(tk[1:])
                    row[6] = lemma
                    row[7] = meaning
                
                row.append(token[bunsetsu]) # BOB

                # LUW
                if L.startswith('B'):
                    if Lpos == pos:
                        row.append('Ba')
                    else:
                        row.append('B')
                    
                    row.append(token.get('l_orthToken', ''))
                    row.append(token.get('l_reading', ''))
                    row.append(token.get('l_lemma', ''))
                    row.append(Lpos) # l_pos
                    row.append(token.get('l_cType', '*'))
                    row.append(token.get('l_cForm', '*'))
                else:
                    row.append('I')
                    row.extend(['*', '*', '*', '*', '*', '*'])
                writer.writerow(row)


@app.command()
def summarize(jsonfiles: List[str]):
    res = dict()
    
    def update(dic1, dic2):
        for k in ("gold", "correct", "system"):
            dic1[k] += dic2[k]

    def recalc(dic):
        for k, d in dic.items():
            d["precision"] = d["correct"] / d["system"]
            d["recall"] = d["correct"] / d["gold"]
            d["f1"] = 2.0 * d["correct"] / (d["system"] + d["gold"])

    for jsonfile in jsonfiles:
        with open(jsonfile) as f:
            js = json.load(f)
            for k, d in js.items():
                if k not in res:
                    res[k] = d
                else:
                    update(res[k], d)
    recalc(res)
    print(json.dumps(res, indent=True))

if __name__ == "__main__":
    app()