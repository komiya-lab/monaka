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

## 推論の実施
```bash
python monaka/cli.py predict-bccwj [inputfile] [outputfile] [model_dirs ...]
```
複数のCVモデルをアンサンブルして推論結果を出力する


## 評価の実施
```bash
python scripts/evaluate.sh {学習に使ったフォルダ} {結果の出力先フォルダ}
```
「結果の出力先フォルダ」に、推論済みの結果とそれに基づく評価結果を出力する。
例: pred.OC_0.eval.txt
```
bunsetsu1(L) count: 10775 correct: 10653 acc: 0.988677494199536
luw(L) count: 10775 correct: 10670 acc: 0.9902552204176334
l_orthToken(L) count: 10775 correct: 10605 acc: 0.9842227378190255
l_reading(L) count: 10775 correct: 10381 acc: 0.9634338747099768
l_pos(L) count: 10775 correct: 10588 acc: 0.9826450116009281
l_cType(L) count: 10775 correct: 10699 acc: 0.9929466357308585
l_cForm(L) count: 10775 correct: 10697 acc: 0.9927610208816705
```
上記は、長単位の推定項目ごとに短単位を1レコードとしての評価結果を提示している。

## 学習のためのTIPS
### Tokenizerの設定
MeCab/fugashiを使うBERTモデル(tohoku-nlpの各種BERTモデルなど)を学習に用いた際は、データセット読み込み時のトークナイザを以下のように設定します。

#### IPAdicを使っているモデル (tohoku-nlp/bert-base-japanese-whole-word-masking など)
```json
{
    "dataeset_options" :{
        "label_file": "",
        "pos_file": "",
        "lm_tokenizer": "bert-tohoku-ja",
        "lm_tokenizer_config": {},
        "pos_as_tokens": false,
        "label_for_all_subwords": false,
        "max_length": {モデルのMAX Token Lengthで書き換え}
    }
}
```

#### UniDic liteを使っているモデル (tohoku-nlp/bert-base-japanese-v2など)
```json
{
    "dataeset_options" :{
        "label_file": "",
        "pos_file": "",
        "lm_tokenizer": "bert-tohoku-ja-unidic",
        "lm_tokenizer_config": {},
        "pos_as_tokens": false,
        "label_for_all_subwords": false,
        "max_length": {モデルのMAX Token Lengthで書き換え}
    }
}
```


#### その他の言語モデル
```json
{
    "dataeset_options" :{
        "label_file": "",
        "pos_file": "",
        "lm_tokenizer": "auto",
        "lm_tokenizer_config": {},
        "pos_as_tokens": false,
        "label_for_all_subwords": false,
        "max_length": {モデルのMAX Token Lengthで書き換え}
    }
}
```