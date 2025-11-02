#! /bin/bash

inputdir=$1

for i in $(seq 0 2)
do
    python monaka/train_cli.py train --device 0 --vocab-dir ${inputdir}/vocab --train-files "${inputdir}/??/train.$i.jsonl" --dev-files "${inputdir}/??/dev.$i.jsonl" --test-files "${inputdir}/??/test.$i.jsonl" config/luw_chunk_bccwj.json exp/bccwj_01/cv_$i
done