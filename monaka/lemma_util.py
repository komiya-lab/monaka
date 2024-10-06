import os
import sys
import csv
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

def parse_bccwj_(path: Path, is_long=False):
    """

    短単位: 
    サブコーパス名	
    サンプルID	
    文字開始位置	原文文字列のサンプル頭からのオフセット値（10きざみ）
    文字終了位置
    連番	サンプル内での長単位の並び順（10きざみ）
    出現形開始位置	書字形出現形のサンプル頭からのオフセット値（10きざみ）
    出現形終了位置
    固定長フラグ	0:固定長でない，1:固定長
    可変長フラグ	0:可変長でない，1:可変長
    文頭ラベル	B:文頭，I:文頭以外
    語彙表ID	書字形出現形のレベルで語を識別するID
    （桁数が大きいためbigint型が必要）
    語彙素ID	UniDicの語彙素を識別するID
    語彙素	短単位情報
    語彙素読み
    語彙素細分類
    語種
    品詞
    活用型
    活用形
    語形
    用法
    書字形
    書字形出現形
    原文文字列
    発音形出現形

    長単位:
    サブコーパス名	
    サンプルID	
    出現形開始位置	書字形出現形のサンプル頭からのオフセット値（10きざみ）
    出現形終了位置
    文節	B:文節，空文字:文節でない
    短長相違フラグ	短単位と長単位の範囲が一致しているかどうか
    0:短長一致，1:短長相違
    固定長フラグ	0:固定長でない，1:固定長
    可変長フラグ	0:可変長でない，1:可変長
    語彙素	長単位情報
    語彙素読み
    語種
    品詞
    活用型
    活用形
    語形
    書字形
    書字形出現形
    原文文字列
    発音形出現形
    連番	サンプル内での長単位の並び順（10きざみ）
    文字開始位置	原文文字列のサンプル頭からのオフセット値（10きざみ）
    文字終了位置
    文頭ラベル	B:文頭，I:文頭でない
    """
    if not is_long:
        fields = ['サブコーパス名', 'サンプルID', '文字開始位置', '文字終了位置', '連番', '出現形開始位置', '出現形終了位置', '固定長フラグ', '可変長フラグ', '文頭ラベル', 
              '語彙表ID', '語彙素ID', '語彙素', '語彙素読み', '語彙素細分類', '語種', '品詞', '活用型', '活用形', '語形', '用法', '書字形', '書字形出現形', '原文文字列', '発音形出現形']
    else:
        fields = ['サブコーパス名', 'サンプルID', '出現形開始位置', '出現形終了位置', '文節', '短長相違フラグ', '固定長フラグ', '可変長フラグ', 
              '語彙素', '語彙素読み', '語種', '品詞', '活用型', '活用形', '語形', '書字形', '書字形出現形', '原文文字列', '発音形出現形', '連番', '文字開始位置', '文字終了位置', '文頭ラベル']

    with open(path) as f:
        res = dict()
        reader = csv.reader(f, delimiter='\t')
        buf = list()
        for row in reader:
            data = dict(zip(fields, row))
            if 'B' in data['文頭ラベル']:
                if len(buf) > 0:
                    head  = buf[0]
                    text = f"{head['サブコーパス名']}_{head['サンプルID']}_{head['文字開始位置']}"
                    #res[text] = ''.join(d['原文文字列'] for d in buf)
                    res[text] = buf
                    buf =list()
            buf.append(data)
        if len(buf) > 0:
            text = f"{head['サブコーパス名']}_{head['サンプルID']}_{head['文字開始位置']}"
            res[text] = buf
    return res


@app.command()
def parse_bccwj(core_suw: Path, core_luw: Path, output: Path):
    suw = parse_bccwj_(core_suw, is_long=False)
    luw = parse_bccwj_(core_luw, is_long=True)

    res = dict()

    print(len(suw))
    print(len(luw))

    for key, dsuw in suw.items():
        if key not in luw:
            print(key)
            continue
        dluw = luw[key]

        for d in dluw:
            #print(d)
            start = int(d['文字開始位置'])
            end = int(d['文字終了位置'])
            suws = [dd for dd in dsuw if int(dd['文字開始位置']) >= start and int(dd['文字開始位置']) < end]
            r = {
                "surface": d['原文文字列'],
                "lemma": d['語彙素'],
                "pos": d['品詞'],
                "suw": [
                    {"surface": s['原文文字列'], "lemma": s['語彙素'], "pos": s['品詞']} for s in suws
                ],
                "text": ''.join([v['原文文字列'] for v in dluw])
            }
            k = to_key(r)
            r['input'] = k
            #res.append(r)
            res[k] = r

    print(len(res))

    with open(output, 'w') as f:
        json.dump(res, f, indent=True, ensure_ascii=False)


def to_pos(tokens: List[str]):
    a = [t for t in tokens[:4] if t!= '*']
    return '-'.join(a)

def parse_cejc_(cabocha_file: Path):
    res = dict()
    buf = list()
    data = None
    with open(cabocha_file) as f:
        for line in f:
            if line.startswith('#') or line.startswith('*'):
                continue
            if line.startswith('EOS'):
                if data is not None:
                    buf.append(data)
                if len(buf) > 0:
                    text = ''.join([d['surface'] for d in buf])
                    for d in buf:
                        d['text'] = text
                        key = to_key(d)
                        d['input'] = key
                        res[key] = d
                    buf = list()
                    data = None
                continue

            line = line.strip()
            try:
                surface, suw, l_surface, luw, bnd = line.split('\t')
            except Exception as e:
                surface, suw, l_surface, luw = line.split('\t')
                #print(line.split('\t'))
                #raise e
            suw_tokens = suw.split(',')
            pos = to_pos(suw_tokens)
            lemma = suw_tokens[7]
            luw_tokens = luw.split(',')
            l_pos = to_pos(luw_tokens)
            if len(l_pos) == 0:
                if data is not None:
                    data['suw'].append({"surface": surface, "pos": pos, "lemma": lemma})
                continue
            l_lemma = luw_tokens[-1]
            if data is not None:
                buf.append(data)
            data = {
                "surface": l_surface,
                "pos": l_pos,
                "lemma": l_lemma,
                "suw": [
                    {"surface": surface, "lemma": lemma, "pos": pos}
                ]
            }

    if len(buf) > 0:
        text = ''.join([d['surface'] for d in buf])
        for d in buf:
            d['text'] = text
            key = to_key(d)
            d['input'] = key
            res[key] = d
            buf.clear()

    return res


@app.command()
def parse_cejc(output: Path, cabocha_files: List[Path]):
    res = dict()
    for fname in cabocha_files:
        res.update(parse_cejc_(fname))

    with open(output, 'w') as f:
        json.dump(res, f, indent=True, ensure_ascii=False)


if __name__ == "__main__":
    app()

    