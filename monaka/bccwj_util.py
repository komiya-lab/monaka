import typer
import json
import csv

from typing import List

app = typer.Typer(pretty_exceptions_show_locals=False)

BCPEXPORT_LIST = [
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

@app.command()
def test():
    print("test")

@app.command()
def export_tsv(base: str, target: str):
    based = dict()
    with open(base) as f:
        for line in f:
            js = json.loads(line)
            based[js["sentence"]] = js

    with open(target) as f:
        for line in f:
            js = json.loads(line)
            name = js["sentence"]
            d = based[name]
            for token in d["tokens"]:
                out = {"corpusName(S)": d["corpusName(S)"], "file(S)": d["file(S)"]}
                out.update(token)
                print("\t".join(out[k] for k in BCPEXPORT_LIST))

@app.command()
def evaluate(gold: str, pred: str, fields: List[str]=["bunsetsu1(L)", "luw(L)", "l_orthToken(L)", "l_reading(L)", "l_pos(L)", "l_cType(L)", "l_cForm(L)"]):
    output = dict()
    with open(gold) as f1:
        grd = csv.reader(f1, delimiter="\t")
        with open(pred) as f2:
            prd = csv.reader(f2, delimiter="\t")
            for rg, rp in zip(grd, prd):
                for field in fields:
                    i = BCPEXPORT_LIST.index(field)
                    d = output.get(field, {"a":0 , "c": 0})
                    d["a"] +=1
                    if field in ["bunsetsu1(L)", "luw(L)"]:
                        if 'B' in rg[i]:
                            if 'B' in rp[i]:
                                d["c"] += 1
                        else:
                            if 'B' not in rp[i]:
                                d["c"] += 1

                    elif rg[i] == rp[i]:
                        d["c"] += 1
                    elif len(rg[i]) == 0 and rp[i] == '*':
                        d["c"] += 1
                    output[field] = d
    
    for field, d in output.items():
        print(f"{field} count: {d['a']} correct: {d['c']} acc: {d['c']/d['a']}")

    
if __name__ == "__main__":
    app()