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


@app.command()
def to_cabocha(datajsonl, unidiccsv: str):
    with open(unidiccsv) as f:
        rd = csv.reader(f)
        unid = dict()
        for row in rd:
            surface = row[0]
            pos = row[4:10]
            lemmaid = row[-1]
            d = unid.get(surface, dict())
            d['-'.join(pos)] = {'rows': row, 'lemmaid':lemmaid, 'pos': pos}
            unid[surface] = d

    with open(datajsonl) as f:
        for i, line in enumerate(f):
            js = json.loads(line)
            pbid = -1
            lines = list()
            lines.append(f'#! DOC {i}')
            lines.append(f'#! DOCATTR	<ID>{i}</ID><sent_id># sent_id = {js["sent_id"]}</sent_id>')
            for token, pos, bid in zip(js['tokens'], js['pos'], js['bid']):
                if bid != pbid:
                    pbid = bid
                    lines.append(f"* {bid} -1D")
                feat = None
                if token in unid:
                    #print(f'token {token} found', file=sys.stderr)

                    pos_tokens = set(pos.split('-'))
                    for key, d in unid[token].items():
                        keyset = set(key.split('-'))
                        if pos_tokens.issubset(keyset):
                            feat = ','.join([r if r != ',' else '，' for r in d['rows'][4:]])
                            break
                if not feat:
                    feat = ','.join(['' for _ in range(29)])
                lines.append(f"{token.strip()}\t{feat}")
            lines.append('EOS')
            print('\n'.join(lines))

@app.command()
def to_cabocha2(datajsonl):

    with open(datajsonl) as f:
        for i, line in enumerate(f):
            js = json.loads(line)
            pbid = -1
            lines = list()
            lines.append(f'#! DOC {i}')
            lines.append(f'#! DOCATTR	<ID>{i}</ID><sent_id># sent_id = {js["sent_id"]}</sent_id>')
            for token, pos, bid in zip(js['tokens'], js['pos'], js['bid']):
                if bid != pbid:
                    pbid = bid
                    dep = js['dependency'][bid]
                    d = dep["head"]
                    if d == bid:
                        d = -1
                    lines.append(f"* {bid} {d}D")
                feat = ','.join(pos.split('-'))
                lines.append(f"{token.strip()}\t{feat}")
            lines.append('EOS')
            print('\n'.join(lines))
                
def chj2unidic(token):
    pos = token['pos(S)'].split('-')
    features = ['' for _ in range(29)]
    for i, p in enumerate(pos):
        features[i] = p
    features[4] = token['sysCType(S)']
    features[5] = token['cForm(S)']
    features[6] = token['reading(S)']
    features[7] = token['lemma(S)']
    features[8] = token['orthToken(S)']
    features[9] = token['pronToken(S)']
    features[10] = token['orthBase(S)']
    features[11] = ""
    features[12] = token['wType(S)']
    features[13] = ""
    features[14] = ""
    features[15] = ""
    features[23] = token['formBase(S)']
    features[27] = token['lid(S)']
    features[28] = token['lemmaID(S)']
    return features
    

@app.command()
def chj2jsonl(chjjsonl: str, bunruifilename: str, bunruihist: str):
    """
    CHJをJSONLに変換したMonaka用データを係り受け解析用JSONLに変換する
    """
    """
    {
  "sent_id": "OC01_00001-1",
  "text": "詰め将棋の本を買ってきました。",
  "bunsetsu": [
    "詰め将棋の",
    "本を",
    "買って",
    "きました。"
  ],
  "pos": [
    "動詞-一般-下一段-マ行",
    "名詞-普通名詞-一般",
    "助詞-格助詞",
    "名詞-普通名詞-一般",
    "助詞-格助詞",
    "動詞-一般-五段-ワア行",
    "助詞-接続助詞",
    "動詞-非自立可能-カ行変格",
    "助動詞-助動詞-マス",
    "助動詞-助動詞-タ",
    "補助記号-句点"
  ],
  "rel": [
    "CONT-compound",
    "SEM_HEAD-shead",
    "SYN_HEAD-case",
    "SEM_HEAD-shead",
    "SYN_HEAD-case",
    "SEM_HEAD-shead",
    "SYN_HEAD-mark",
    "ROOT-shead",
    "SYN_HEAD-aux",
    "FUNC-aux",
    "CONT-punct"
  ],
  "tokens": [
    "詰め",
    "将棋",
    "の",
    "本",
    "を",
    "買っ",
    "て",
    "き",
    "まし",
    "た",
    "。"
  ],
  "bid": [
    0,
    0,
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    3,
    3
  ],
  "dependency": [
    {
      "head": 1,
      "rel": "nmod",
      "id": 0
    },
    {
      "head": 2,
      "rel": "obj",
      "id": 1
    },
    {
      "head": 3,
      "rel": "advcl",
      "id": 2
    },
    {
      "head": 3,
      "rel": "root",
      "id": 3
    }
  ],
  "lemma": [
    "詰める",
    "将棋",
    "の",
    "本",
    "を",
    "買う",
    "て",
    "来る",
    "ます",
    "た",
    "。"
  ],
  "misc": [
    {
      "BunsetuBILabel": "B",
      "BunsetuPositionType": "CONT",
      "LUWBILabel": "B",
      "LUWPOS": "名詞-普通名詞-一般",
      "SpaceAfter": "No",
      "UnidicInfo": "ツメル,詰める,詰め,詰める,ツメ,,,ツメル,ツメショウギ,詰め将棋"
    },
  ],
  "upos": [
    "VERB",
    "NOUN",
    "ADP",
    "NOUN",
    "ADP",
    "VERB",
    "SCONJ",
    "VERB",
    "AUX",
    "AUX",
    "PUNCT"
  ]
}
    """
    """
    {
  "sentence": "やまとうたは、人の心を種として、万の言の葉とぞなれりける。",
  "tokens": [
    {
      "start(S)": "10",
      "end(S)": "40",
      "boundary(S)": "B",
      "orthToken(S)": "やまと",
      "pronToken(S)": "ヤマト",
      "reading(S)": "ヤマト",
      "lemma(S)": "ヤマト",
      "originalText(S)": "やまと",
      "pos(S)": "名詞-固有名詞-地名-一般",
      "sysCType(S)": "",
      "cForm(S)": "",
      "apply(S)": "",
      "additionalInfo(S)": "",
      "lid(S)": "10572912436322816",
      "meaning(S)": "",
      "UpdUser(S)": "ymatuzaki",
      "UpdDate(S)": "2023-03-09 13:35:53.417",
      "order(S)": "10",
      "note(S)": "",
      "open(S)": "10",
      "close(S)": "40",
      "wType(S)": "固",
      "fix(S)": "0",
      "variable(S)": "1",
      "formBase(S)": "ヤマト",
      "lemmaID(S)": "38464",
      "usage(S)": "",
      "sentenceId(S)": "10",
      "s_memo(S)": "",
      "origChar(S)": "やまと",
      "pSampleID(S)": "0",
      "pStart(S)": "0",
      "orthBase(S)": "やまと",
      "file(L)": "1101_古今和歌集_S001_仮名序",
      "l_orthToken(L)": "やまとうた",
      "l_pos(L)": "名詞-普通名詞-一般",
      "l_cType(L)": "",
      "l_cForm(L)": "",
      "l_reading(L)": "ヤマトウタ",
      "l_lemma(L)": "やまと歌",
      "luw(L)": "B",
      "memo(L)": "",
      "UpdUser(L)": "ymatuzaki",
      "UpdDate(L)": "2023-03-09 13:46:57.903",
      "l_start(L)": "10",
      "l_end(L)": "60",
      "bunsetsu1(L)": "B",
      "bunsetsu2(L)": "",
      "corpusName(L)": "CHJ平安",
      "diffSuw(L)": "1",
      "l_lemmaNew(L)": "やまと歌",
      "l_readingNew(L)": "ヤマトウタ",
      "l_orthBase(L)": "やまとうた",
      "l_formBase(L)": "ヤマトウタ",
      "l_pronToken(L)": "ヤマトウタ",
      "l_wType(L)": "混",
      "l_originalText(L)": "やまとうた",
      "complex(L)": "0",
      "l_meaning(L)": "",
      "l_kanaToken(L)": "ヤマトウタ",
      "l_formOrthBase(L)": "やまと歌",
      "l_origChar(L)": "やまとうた",
      "note(L)": "",
      "pSampleID(L)": "0",
      "pStart(L)": "0",
      "rn": "1"
    },

  "corpusName(S)": "CHJ平安",
  "file(S)": "1101_古今和歌集_S001_仮名序"
    """
    with open(bunruifilename) as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd) # skip header
        lemmaid2wlsp = {row[1].strip(): row[0].split(',')[0] for row in rd if len(row) > 1}
    
    with open(bunruihist) as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd) # skip header
        d2 = {row[1].strip(): f"{row[0][0]}.{row[0].strip()[1:]}" for row in rd if len(row) > 1}
        lemmaid2wlsp.update(d2)
    
    with open(chjjsonl) as f:
        for line in f:
            js = json.loads(line)
            tokens = [t for t in js['tokens'] if t['pos(S)'] not in ['空白', '記号-空白']]
            res = {
                "sent_id": f"{js['corpusName(S)']}_{js['file(S)']}_{js['tokens'][0]['start(S)']}",
                "text": js['sentence'],
                "bunsetsu": [],
                "tokens": [token['origChar(S)'] for token in tokens],
                "bid": [],
                "unidic": [],
                "wlsp": None,
                "misc": [],
                "lemma": [token['lemma(S)'] for token in tokens],
                "lid": [token['lemmaID(S)'] for token in tokens],
                "dependency": [],
                "pos": [f"{token['pos(S)']}-{token['sysCType(S)']}" if len(token['sysCType(S)']) > 2 else token['pos(S)'] for token in tokens],
            }
            bid = -1
            bnst = None
            for token in tokens:
                flg = 'B' in token['bunsetsu1(L)']
                if bnst:
                    if flg:
                        res['bunsetsu'].append(bnst)
                        bnst = ''
                        bid += 1
                else:
                    bid = 0
                res['bid'].append(bid)
                res['unidic'].append(chj2unidic(token))

                if bnst is None:
                    bnst = token['origChar(S)']
                else:
                    bnst += token['origChar(S)']
            
            if bnst is None:
                continue
            if len(bnst) > 0:
                res['bunsetsu'].append(bnst)

            res['wlsp'] = [lemmaid2wlsp.get(l, '') for l in res['lid']]

            print(json.dumps(res, ensure_ascii=False))



def read_cabocha(cabochafile:str):
    with open(cabochafile) as f:
        dep = list()
        for line in f:
            if 'EOS' in line:
                yield dep
                dep.clear()
            if line.startswith('*'):
                tokens = line.split(' ')
                id_ = int(tokens[1])
                head = int(tokens[2][:-1]) # nD形式なのでD削除
                if head == -1:
                    head = id_
                dep.append({'id': id_, 'head': head})
        

@app.command()
def cabocha2jsonl(datajsonl:str, cabochafile:str):
    with open(datajsonl) as f:
        for line, cdep in zip(f, read_cabocha(cabochafile)):
            js = json.loads(line)
            for d in cdep:
                js['dependency'][d['id']]['head'] = d['head']
            
            print(json.dumps(js, ensure_ascii=False))



@app.command()
def analyze_pred(goldjsonl: str, predjsonl: str):
    distacc = dict()
    relacc = dict()
    with open(goldjsonl) as fg, open(predjsonl) as fp:
        for lg, lp in zip(fg, fp):
            gold = json.loads(lg)
            pred = json.loads(lp)

            for gdep, pdep in zip(gold['dependency'], pred['dependency']):
                dist = gdep['head'] - gdep['id']
                rel = gdep['rel']
                dd = distacc.get(dist, {'a': 0, 'c': 0})
                dr = relacc.get(rel, {'a': 0, 'c': 0})
                dd['a'] += 1
                dr['a'] += 1
                if pdep['head'] == gdep['head']:
                    dd['c'] += 1
                    dr['c'] += 1

                distacc[dist] = dd
                relacc[rel] = dr
    
    for d in relacc.values():
        d['acc'] = d['c'] /d ['a']
    
    min_dist = min(distacc.keys())
    max_dist = max(distacc.keys())
    distances = list(range(min_dist, max_dist+1))
    res = {
        'distances': distances, 
        'dist_count': [distacc[d]['a'] if d in distacc else 0 for d in distances],
        'dist_correct': [distacc[d]['c'] if d in distacc else 0 for d in distances],
        'dist_acc': [distacc[d]['c'] / distacc[d]['a'] if d in distacc else None for d in distances],
        'rel': relacc
    }

    print(json.dumps(res, indent=True))


if __name__ == "__main__":
    app()
