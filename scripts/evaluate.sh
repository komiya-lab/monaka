#! /bin/bash

inputdir=$1
outputdir=$2

for reg in $(echo OC OW OY PB PM PN)
do
    for c in $(seq 0 2)
    do
        python monaka/cli.py predict-bccwj --device 0 --input-format bcpexport --output-format bcpexport ${inputdir}/${reg}/test.${c}.tsv ${outputdir}/pred.${reg}_${c}.tsv exp/bccwj_01/cv_${c}/
        python monaka/bccwj_util.py evaluate ${inputdir}/${reg}/test.${c}.tsv ${outputdir}/pred.${reg}_${c}.tsv > ${outputdir}/pred.${reg}_${c}.eval.txt
    done 
done
