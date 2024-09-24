import os
import sys
import typer
import json
import numpy as np

from pathlib import Path
from typing import List, Optional

app = typer.Typer(pretty_exceptions_show_locals=False)

def parse(js):
    luw = list()
    buf = None
    for token in js["tokens"]:
        begin = token["luw(L)"]
        if 'B' in begin:
            if buf is not None:
                luw.append(buf)
            buf = {
                "surface": token["l_orthToken(L)"],
                "lemma": token["l_lemma(L)"],
                "pos": token["l_pos(L)"],
                "suw": [
                    {"pos": token["pos(S)"], "lemma": token["lemma(S)"], "surface": token["orthToken(S)"]}
                ]
            }
        else:
            try:
                buf["suw"].append({"pos": token["pos(S)"], "lemma": token["lemma(S)"], "surface": token["orthToken(S)"]})
            except:
                pass
    if buf is not None:
        luw.append(buf)
    return luw

def to_key(obj):
    return f"pos: {obj['pos']} surface: {obj['surface']} suw_pos: {' '.join([d['pos'] for d in obj['suw']])} suw_surface: {' '.join([d['surface'] for d in obj['suw']])} suw_lemma: {' '.join([d['lemma'] for d in obj['suw']])}"

@app.command()
def extract_lemma_info(jsonlfile: Path, outfile: Path):
    luws = dict()
    with open(jsonlfile) as f:
        for line in f:
            js = json.loads(line)
            l = parse(js)
            for obj in l:
                key = to_key(obj)
                obj['input'] = key
                if key in luws:
                    if obj['lemma'] != luws[key]['lemma']:
                        print(f"unmatched lemma: {obj['lemma']} {luws[key]['lemma']}, key: {key}", file=sys.stderr)
                        continue
                luws[key] = obj

    with open(outfile, 'w') as f:
        json.dump(luws, f, ensure_ascii=False, indent=True)

if __name__ == "__main__":
    app()

    