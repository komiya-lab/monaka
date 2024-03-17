import os
import typer
import json


app = typer.Typer()
def load_chj(fname: str):
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
    # generatorにすることでメモリを節約
    with open(fname) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            yield dict(zip(head, row))

def to_sentences(data):
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
def chj2jsonl(fname: str):
    for d in to_sentences(load_chj(fname)):
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
    return f'{data.get("pos(S)", "")}_{data.get("sysCType(S)", "")}_{data.get("cForm(S)", "")}'

def chjlpos(data: dict):
    return f'{data.get("l_pos(L)", "")}_{data.get("l_CType(L)", "")}_{data.get("l_cForm(L)", "")}'

@app.command()
def chjjsonl2luwjson(jsonlfile: str, luw: bool=True, chunk: bool=True):
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
                    tag += chjlpos(token)
                res['labels'].append(tag)
            print(json.dumps(res, ensure_ascii=False))

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

if __name__ == "__main__":
    app()