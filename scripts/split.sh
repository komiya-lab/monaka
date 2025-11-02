#! /bin/bash

fname=$1
outputdir=$2


for reg in $(echo OC OW OY PB PM PN)
do
    cat ${fname} | grep BWJ_${reg}_core > ${fname}.${reg}
    python util.py chjjsonl2luwjson ${fname}.${reg} > ${fname}.${reg}.jsonl
    python monaka/train_cli.py create-split --folds 3 --dev-ratio 0.1 --test-ratio 0.1 ${outputdir}/${reg} ${fname}.${reg}.jsonl

done

for i in $(seq 0 2)
do
    cat $outputdir/*/train.$i.jsonl > $outputdir/train.$i.jsonl
done