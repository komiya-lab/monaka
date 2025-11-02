# BCCWJ2構築用ブランチ


## データ作成
以下の手順でBCCWJのダンプデータをJSON-L形式に変換する。
```bash
export PYTHONPATH=[monakaライブラリの場所]
python util.py chj2jsonl [ダンプデータ] > [出力先JOSN-Lファイル]
```

### BCCWJ LUW コア統計情報 (検証データ作成用)
| レジスタ | 文数 |
|---------|-----|
| OC | 6412 |
| OW | 6037 |
| OY | 7455 |
| PB | 10075 |
| PM | 12953 |
| PN| 17051 |

このため、各レジスタから10%程度、学習・検証データを抽出する。Foldは検証の効率化のために3とする。


### スプリットの作成
以下のようなシェルスクリプトでスプリットを作成する (scripts/split.shを参照)
なお、util.pyの chjjsonl2luwjson コマンドは、BCCWJコアダンプデータをJSON-L化したものから、Monakaが学習に使用するファイルに変換するスクリプト

```bash

fname=$1
outputdir=$2


for reg in $(echo OC OW OY PB PM PN)
do
    cat ${fname} | grep BWJ_${reg}_core > ${fname}.${reg}
    python util.py chjjsonl2luwjson ${fname}.${reg} > ${fname}.${reg}.jsonl
    python monaka/train_cli.py create-split --folds 3 --dev-ratio 0.1 --test-ratio 0.1 ${outputdir}/${reg} ${fname/txt/jsonl}.${reg}

done

for i in $(seq 0 2)
do
    cat $outputdir/*/train.$i.jsonl > $outputdir/train.$i.jsonl
done

```
### vocabファイルの作成
学習対象のラベルと品詞情報のIDをデータベース化する。

```bash
python monaka/train_cli.py create-vocab [出力先] [入力JSON-Lファイル]
```

## 学習の実施
scripts/train_bccwj.sh を参考に学習を実施

```bash
#! /bin/bash

inputdir=$1

for i in $(seq 0 2)
do
    python monaka/train_cli.py train --device 0 --vocab-dir ${inputdir}/vocab --train-files "${inputdir}/??/train.$i.jsonl" --dev-files "${inputdir}/??/dev.$i.jsonl" --test-files "${inputdir}/??/test.$i.jsonl" config/luw_chunk_bccwj.json exp/bccwj_01/cv_$i
done
```