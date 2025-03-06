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
                            drel[syn_head] = 'shead'
                        syn_head = -1
                bids.append(bid)

                pos.append(token['xpos'])
                buf.append(token['form'])

                bunsetsu_heads[token['id']] = len(bunsetsu) 
                if 'SEM_HEAD' in ttype or 'ROOT' in ttype:
                    bunsetsu_deps.append(token['head'])
                    bunsetsu_deprel.append(token['deprel'])
                    drel.append('shead')
                    head_exists = True
                else:
                    if ('SYN_HEAD' in ttype or token['head'] > len(bids)) and syn_head < 0:
                        syn_head = token['id'] -1
                    drel.append(token['deprel'])
            
            if len(buf) > 0:
                    bunsetsu.append(''.join(buf))
                    if not head_exists:
                        h = sent[syn_head]
                        bunsetsu_deps.append(h['head'])
                        bunsetsu_deprel.append(h['deprel'])
                        drel[syn_head] = 'shead'

            res['bunsetsu'] = bunsetsu
            res['pos'] = pos
            res['rel'] = drel
            res['tokens'] = tokens
            res['bid'] = bids
            res['dependency'] = list()

            for i, (head, rel) in enumerate(zip(bunsetsu_deps, bunsetsu_deprel)):
                 if head == 0:
                    res['dependency'].append({"head": i, "rel": rel, 'id': i})
                    continue
                 j = bunsetsu_heads[head]
                 res['dependency'].append({"head": j, "rel": rel, "id": i})

            print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    app()
