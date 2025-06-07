import os
import sys
import csv
import typer
import json
import numpy as np
from conllu import parse_incr

from pathlib import Path
from typing import List, Optional

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def conllu2dep(fname: str):
    with open(fname) as f:
        for sent in parse_incr(f):
            res = dict()
            res.update(sent.metadata)
            bunsetsu = list()
            poss = list()
            bunsetsu_heads = dict()
            bunsetsu_deps= list()
            bunsetsu_deprel = list()
            drel = list()
            buf = list()
            tokens = list()
            pos = list()
            bid = 0
            bids = list()
            syn_head = -1
            head_exists = False

            for token in sent:
                ttype = token['misc']['BunsetuPositionType']
                b = token['misc']['BunsetuBILabel']
                tokens.append(token['form'])
                if 'B' in b:
                    if len(buf) > 0:
                        bid += 1
                        bunsetsu.append(''.join(buf))
                        buf.clear()
                        if head_exists:
                            head_exists = False
                        else:
                            if syn_head < 0:
                                syn_head = 0
                            h = sent[syn_head]
                            print(sent.metadata, file=sys.stderr)
                            print(bunsetsu[-1], file=sys.stderr)
                            bunsetsu_deps.append(h['head'])
                            bunsetsu_deprel.append(h['deprel'])
                            drel[syn_head] = f'{ttype}-shead'
                        syn_head = -1
                bids.append(bid)

                pos.append(token['xpos'])
                buf.append(token['form'])

                bunsetsu_heads[token['id']] = len(bunsetsu) 
                if 'SEM_HEAD' in ttype or 'ROOT' in ttype:
                    bunsetsu_deps.append(token['head'])
                    bunsetsu_deprel.append(token['deprel'])
                    drel.append(f'{ttype}-shead')
                    head_exists = True
                else:
                    if ('SYN_HEAD' in ttype or token['head'] > len(bids)) and syn_head < 0:
                        syn_head = token['id'] -1
                    drel.append(f"{ttype}-{token['deprel']}")
            
            if len(buf) > 0:
                    bunsetsu.append(''.join(buf))
                    if not head_exists:
                        h = sent[syn_head]
                        bunsetsu_deps.append(h['head'])
                        bunsetsu_deprel.append(h['deprel'])
                        drel[syn_head] = f'{ttype}-shead'

            res['bunsetsu'] = bunsetsu
            res['pos'] = pos
            res['rel'] = drel
            res['tokens'] = tokens
            res['bid'] = bids
            res['dependency'] = list()
            res['lemma'] = [token['lemma'] for token in sent]
            res['misc'] = [token['misc'] for token in sent]
            res['upos'] = [token['upos'] for token in sent]

            for i, (head, rel) in enumerate(zip(bunsetsu_deps, bunsetsu_deprel)):
                 if head == 0:
                    res['dependency'].append({"head": i, "rel": rel, 'id': i})
                    continue
                 j = bunsetsu_heads[head]
                 res['dependency'].append({"head": j, "rel": rel, "id": i})

            print(json.dumps(res, ensure_ascii=False))

@app.command()
def rulecheck(conllufiles: List[Path]):
    rules = dict()
    for fname in conllufiles:
        with open(fname) as f:
            for sent in parse_incr(f):
                bid = -1
                bids = list()
                for token in sent:
                    b = token['misc']['BunsetuBILabel']
                    if b.startswith('B'):
                        bid += 1
                    bids.append(bid)
                for token in sent:
                    rel = token['deprel']
                    ttype = token['misc']['BunsetuPositionType']
                    rule = rules.get(f'{ttype}-{rel}', {'SYN_HEAD': 0, 'root': 0, 'next': 0, 'prev': 0, 'outside': 0, 'others': 0})
                    head = token['head']
                    id_ = token['id']
                    htype = sent[head-1]['misc']['BunsetuPositionType']
                    bid = bids[id_ -1]
                    hid = bids[head -1]
                    if head == 0:
                        rule['root'] += 1
                    elif bid != hid:
                        rule['outside'] += 1
                    elif htype in ('ROOT', 'SEM_HEAD'):
                        rule['SYN_HEAD'] += 1
                    elif id_ - head == 1:
                        rule['prev'] += 1
                    elif head - id_ == 1:
                        rule['next'] += 1
                    else:
                        rule['others'] += 1
                    
                    rules[f'{ttype}-{rel}'] = rule
    print(json.dumps(rules, indent=True))


@app.command()
def core2unidic(corefilename: str, bunruifilename: str):
    """
    BCCWJ TSV
    ----------
    0 サブコーパス名	
    1 サンプルID	
    2 文字開始位置	原文文字列のサンプル頭からのオフセット値（10きざみ）
    3 文字終了位置
    4 連番	サンプル内での長単位の並び順（10きざみ）
    5 出現形開始位置	書字形出現形のサンプル頭からのオフセット値（10きざみ）
    6 出現形終了位置
    7 固定長フラグ	0:固定長でない，1:固定長
    8 可変長フラグ	0:可変長でない，1:可変長
    9 文頭ラベル	B:文頭，I:文頭以外
    10 語彙表ID	書字形出現形のレベルで語を識別するID
    （桁数が大きいためbigint型が必要）
    11 語彙素ID	UniDicの語彙素を識別するID
    12 語彙素	短単位情報
    13 語彙素読み
    14 語彙素細分類
    15 語種
    16 品詞
    17 活用型
    18 活用形
    19 語形
    20 用法
    21 書字形
    22 書字形出現形
    23 原文文字列
    24 発音形出現形
    ----------
    
    UniDic
    ----------
    4	%f[0]	pos1	品詞大分類	名詞	動詞
    5	%f[1]	pos2	品詞中分類	固有名詞	一般
    6	%f[2]	pos3	品詞小分類	人名	＊
    7	%f[3]	pos4	品詞細分類	名	＊
    8	%f[4]	cType	活用型	＊	下一段-ラ行
    9	%f[5]	cForm	活用形	＊	未然形-一般
    10	%f[6]	lForm	語彙素読み	モモタロウ	ハシル
    11	%f[7]	lemma	語彙素	モモタロウ	走る
    12	%f[8]	orth	書字形出現形	桃太郎	走れ
    13	%f[9]	pron	発音形出現形	モモタロー	ハシレ
    14	%f[10]	orthBase	書字形基本形	桃太郎	走れる
    15	%f[11]	pronBase	発音形基本形	モモタロー	ハシレル
    16	%f[12]	goshu	語種	固	和
    17	%f[13]	iType	語頭変化型	＊	＊
    18	%f[14]	iForm	語頭変化形	＊	＊
    19	%f[15]	fType	語末変化型	＊	＊
    20	%f[16]	fForm	語末変化形	＊	＊
    21	%f[17]	iConType	語頭変化結合形	＊	＊
    22	%f[18]	fConType	語末変化結合形	＊	＊
    23	%f[19]	type		名	用
    24	%f[20]	kana	仮名形出現形	モモタロウ	ハシレ
    25	%f[21]	kanaBase	仮名形基本形	モモタロウ	ハシレル
    26	%f[22]	form	語形出現形	モモタロウ	ハシレ
    27	%f[23]	formBase	語形基本形	モモタロウ	ハシレル
    28	%f[24]	aType		2	3
    29	%f[25]	aConType		＊	C1
    30	%f[26]	aModType		＊	M4@1
    31	%f[27]	lid	語彙表ID	2018...	8167...
    32	%f[28]	lemmaID	語彙素ID	73439	29712

    """
    with open(bunruifilename) as f:
        rd = csv.reader(f, delimiter='\t')
        lemmaid2wlsp = {row[1].strip(): row[0].split(',')[0] for row in rd}

    with open(corefilename) as f:
        rd = csv.reader(f, delimiter='\t')
        ubuf = list()
        wbuf = list()
        tokens = list()
        pointer = 0
        pbuf = list()
        sampleid = None

        for row in rd:
            if sampleid != row[1]:
                if len(ubuf) > 0:
                    print(json.dumps({"sample_id": sampleid, 'unidic': ubuf, 'wlsp': wbuf, 'pointers': pbuf, 'sentence': ''.join(tokens)} , ensure_ascii=False))
                    ubuf.clear()
                    wbuf.clear()
                    tokens.clear()
                    pointer = 0

            pos = row[16]
            sampleid = row[1]
            pbuf.append(pointer)
            l = int((int(row[3]) - int(row[2])) / 10)
            pointer += l
            tokens.append(row[23][:l])

            undc = ['' for _ in range(29)]
            # pos
            for i, p in enumerate(pos.split('-')):
                undc[i] = p
            undc[4] = row[17] # 活用型
            undc[5] = row[18] # 活用形
            undc[8] = row[22] # 書字形出現形
            undc[10] = row[21] # 書字形 ? 書字形基本形
            undc[9] = row[24] # 発音形出現形
            undc[7] = row[12] # 語彙素
            undc[23] = row[19] # 語形 ? 語形基本形
            undc[27] = row[10] #lid
            undc[28] = row[11] # lemma id
            undc[12] = row[15] # 語種

            ubuf.append(undc)
            wbuf.append(lemmaid2wlsp.get(row[11], ''))

        
        if len(ubuf) > 0:
            print(json.dumps({"sample_id": sampleid, 'unidic': ubuf, 'wlsp': wbuf, 'pointers': pbuf, 'sentence': ''.join(tokens)} , ensure_ascii=False))


@app.command()
def depWithCore(corejsonl: str, depjsonl: str):
    cdic = dict()

    with open(corejsonl) as f:
        for line in f:
            js = json.loads(line)
            cdic[js['sample_id']] = js
            
    with open(depjsonl) as f:
        for line in f:
            js = json.loads(line)
            tsid = js['sent_id']
            sample_id = "_".join(tsid.split("_")[-2:]).split("-")[0]

            if sample_id not in cdic:
                print(sample_id, tsid, file=sys.stderr)
                break

            samples = cdic[sample_id]
            text = js['text']
            start = samples['sentence'].find(text)
            if start < 0:
                print(js['text'], '\n', samples['sentence'], file=sys.stderr)
                break
            ind = samples['pointers'].index(start)
            undic = list()
            wlsp = list()
            for t in js['tokens']:
                while samples['unidic'][ind][0] == '空白':
                    ind += 1
                undic.append(samples['unidic'][ind])
                wlsp.append(samples['wlsp'][ind])
                ind += 1


            js['unidic'] = undic
            js['wlsp'] = wlsp
            print(json.dumps(js, ensure_ascii=False))


@app.command()
def wlsp(datajsonl: str, bunruifilename: str, bunruihist: str, unidiccsv: str):
    with open(bunruifilename) as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd) # skip header
        lemmaid2wlsp = {row[1].strip(): row[0].split(',')[0] for row in rd if len(row) > 1}
    
    with open(bunruihist) as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd) # skip header
        d2 = {row[1].strip(): f"{row[0][0]}.{row[0].strip()[1:]}" for row in rd if len(row) > 1}
        lemmaid2wlsp.update(d2)
    
    with open('lemma2wlsp.json', 'w') as f:
        json.dump(lemmaid2wlsp, f, ensure_ascii=False, indent=True)

    print(len(lemmaid2wlsp), file=sys.stderr)

    with open(unidiccsv) as f:
        rd = csv.reader(f)
        unid = dict()
        for row in rd:
            surface = row[0]
            pos = row[4:10]
            lemmaid = row[-1]
            d = unid.get(surface, dict())
            d['-'.join(pos)] = {'rows': row, 'lemmaid':lemmaid, 'wlsp': lemmaid2wlsp.get(lemmaid, ''), 'pos': pos}
            unid[surface] = d

    with open(datajsonl) as f:
        for line in f:
            js = json.loads(line)
            wlps = list()

            assert(len(js['tokens']) == len(js['pos']))
            for token, pos in zip(js['tokens'], js['pos']):

                flg = False
                if token in unid:
                    #print(f'token {token} found', file=sys.stderr)

                    pos_tokens = set(pos.split('-'))
                    for key, d in unid[token].items():
                        keyset = set(key.split('-'))
                        if pos_tokens.issubset(keyset):
                            wlps.append(d['wlsp'])
                            flg = True
                            break
                if not flg:
                    wlps.append('')
            js['wlsp'] = wlps
            assert len(js['wlsp']) == len(js['pos']), f"{len(js['wlsp'])}, {len(js['pos'])}" 
            print(json.dumps(js, ensure_ascii=False))


@app.command()
def wlsp_dict(bunruifilename: str):
    d = {"unk": 0, "pad": 1}
    with open(bunruifilename) as f:
        rd = csv.reader(f, delimiter='\t')
        for row in rd:
            wid = row[0].split(',')[0]
            if wid not in d:
                d[wid] = len(d)
    print(json.dumps(d, indent=True, ensure_ascii=False))


@app.command()
def score(goldjsonl:str, predjsonl:str):
    a = 0
    b = 0
    c = 0
    with open(goldjsonl) as fg, open(predjsonl) as fp:
        for lg, lp in zip(fg, fp):
            gold = json.loads(lg)
            pred = json.loads(lp)
            c += len(gold['dependency'])
            for dg, dp in zip(gold['dependency'], pred['dependency']):
                if dg['head'] == dp['head']:
                    a += 1
                    if dg['rel'] == dp['rel']:
                        b += 1
    res = {'all': c, 'u_correct': a, 'l_correct': b, 'u_acc': a/c, 'l_acc': b/c}
    print(json.dumps(res, indent=True))





if __name__ == "__main__":
    app()
