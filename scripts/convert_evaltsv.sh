#! /bin/bash

target=$1

for reg in $(echo OC OW OY PB PM PN)
do
    python monaka/bccwj_util.py export-tsv ../data/proc/20251031_BCCWJ1coreLUW_bcpExport.jsonl.${reg} $target/${reg}/test.0.jsonl > $target/${reg}/test.0.tsv
    python monaka/bccwj_util.py export-tsv ../data/proc/20251031_BCCWJ1coreLUW_bcpExport.jsonl.${reg} $target/${reg}/test.0.jsonl > $target/${reg}/test.1.tsv
    python monaka/bccwj_util.py export-tsv ../data/proc/20251031_BCCWJ1coreLUW_bcpExport.jsonl.${reg} $target/${reg}/test.0.jsonl > $target/${reg}/test.2.tsv
done
