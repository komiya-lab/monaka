# Monaka
A Japanese parser (including support for historical Japanese)

## Installation

## Parse

First, download  and install appropriate UniDic dictionary:
```sh
monaka download wabun
```

Available dictionaries: (x/y means y is alternative of x.)
| name | discription |
|------|-------------|
| gendai | 現代書き言葉 |
| spoken/unidic-spoken | 現代話し言葉 |
| novel/65_novel | 近現代口語小説 |
| qkana/60b_qkana | 旧仮名口語 |
| kindai/60a_kindai-bungo" | 近代文語 |
| 50a_kinsei-bungo" | 近世文語 |
| 50c_kinsei-edo" | 近世江戸 |
| 50b_kinsei-kamigata" | 近世上方 |
| kinsei | 近世江戸口語 |
| kyogen/40_chusei-kougo | 中世口語 |
| wakan/30_chusei-bungo | 中世文語 |
| wabun/20_chuko | 中古和文 |
| manyo/10_jodai | 上代語 |
| 70_waka | 和歌 |
| 80_kansai_hougen | 関西方言 |

Then, call parse command:
```sh
monaka parse {model} 今日はいい天気ですね
```

output:
```json
{
  "tokens": [
    "今日",
    "は",
    "いい",
    "天気",
    "です",
    "ね"
  ],
  "pos": [
    "名詞-普通名詞-副詞可能",
    "助詞-係助詞",
    "形容詞-非自立可能-形容詞",
    "名詞-普通名詞-一般",
    "助動詞-助動詞-デス",
    "助詞-終助詞"
  ],
  "luw": [
    "名詞-普通名詞-一般",
    "助詞-係助詞",
    "形容詞-一般-形容詞",
    "名詞-普通名詞-一般",
    "助動詞-助動詞-デス",
    "助詞-終助詞"
  ],
  "chunk": [
    "B",
    "I",
    "B",
    "B",
    "I",
    "I"
  ],
  "sentence": "今日はいい天気ですね"
}
```

You can specify output format ("bunsetsu-split" and "luw-split" )

```sh
monaka parse {model} 今日はいい天気ですね --output-format bunsetu-split

今日は いい 天気ですね
```

You can also specify MeCab compatible format:
```sh
monaka parse {model} 今日はいい天気ですね --output-format mecab

今日    キョー  キョウ  今日    名詞-普通名詞-副詞可能  *       *       *       名詞-普通名詞-一般      B
は      ワ      ハ      は      助詞-係助詞     *       *       *       助詞-係助詞     I
いい    イー    ヨイ    良い    形容詞-非自立可能       形容詞  連体形-一般     *       形容詞-一般-形容詞      B
天気    テンキ  テンキ  天気    名詞-普通名詞-一般      *       *       *       名詞-普通名詞-一般      B
です    デス    デス    です    助動詞  助動詞-デス     終止形-一般     *       助動詞-助動詞-デス      I
ね      ネ      ネ      ね      助詞-終助詞     *       *       *       助詞-終助詞     I
EOS
```

You can use MeCab formatting strings:
```sh
monaka parse {model} 今日はいい天気ですね --output-format mecab --node-format "%m,%f[0,1,2],%l,%b\n" --eos-format "\n"

今日,名詞,普通名詞,副詞可能,名詞-普通名詞-一般,B
は,助詞,係助詞,*,助詞-係助詞,I
いい,形容詞,非自立可能,*,形容詞-一般-形容詞,B
天気,名詞,普通名詞,一般,名詞-普通名詞-一般,B
です,助動詞,*,*,助動詞-助動詞-デス,I
ね,助詞,終助詞,*,助詞-終助詞,I
```

| Format | Detail |
|--------|--------|
| %b     | Bunsetsu boundary flag |
| %l     | LUW PoS tag |

## Server
You can start parsing server
```sh
monaka_server

 * Serving Flask app 'monaka.server'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

Parse service URL: POST <server>/model/<model>/dic/<dic>/parse

Example:
```
http://127.0.0.1:5000/model/all_in_one/dic/wabun/parse

{
  "sentence": ["今は昔竹取の翁といふものありけり"],
  "node_format": "%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[13]\t%f[26]\t%f[27]\t%f[28]\n",
  "output_format": "mecab",
  "unk_format": "%m\t%m\t%m\t%m\tUNK\t%f[4]\t%f[5]\t\n",
  "eos_format": "EOS\n",
  "bos_format": ""
}

```

Please see scripts/monaka_client.sh for more detail.


## Model download

The author will provide trained model upon a request. Please contact the author.

## Training monaka model

### LUW and Bunsetsu tokenizer/chunker

#### Creating Dataset

A dataset should be JSON-L formatted and its each line shoud contains following fields:
```json
    {
        "sentence": "str", 
        "tokens": ["a list of SUW"],
        "pos": ["POS-tag labels for each SUW"],
        "labels": ["Target labels for each SUW"]
    }
```

We provide data conversion script for UD-Japanese data.
Here is an example command to convert UD-Japanese-GSD train data.

```sh
monaka_train ud2jsonl ja_gsd-ud-train.conllu ja_gsd-ud-train.jsonl
```

After creating the dataset files, then create label and pos-tag dictionaries:

```sh
monaka_train create-vocab [output_dir] ja_gsd-ud-train.jsonl ja_gsd-ud-dev.jsonl ja_gsd-ud-test.jsonl
```

#### Creating training configuration JSON file
An example configuration is in config/luw_chunk_base.config.json.

```json
{
    "train_files": [""],
    "dev_files": [""],
    "test_files": null,
    "dataeset_options" :{
        "label_file": "vocab/labels.json",
        "pos_file": "vocab/pos.json",
        "lm_tokenizer": "bert-tohoku-ja",
        "lm_tokenizer_config": {},
        "pos_as_tokens": false,
        "label_for_all_subwords": true,
        "max_length": 512
    },
    "model_name": "SeqTagging",
    "model_config": {
        "n_pos_emb": 256,
        "pos_dropout": 0.5,
        "mlp_dropout": 0.5,
        "lm_class_name": "AutoLM",
        "lm_class_config": {
            "model": "cl-tohoku/bert-base-japanese-whole-word-masking",
            "requires_grad": true,
            "use_scalar_mix": false,
            "sclar_mix_dropout": 0.5,
            "use_attentions": false
        },
        "pos_padding_idx": 1
    },
    "batch_size": 32,
    "epochs": 50,
    "lr": 2e-5,
    "decay": 0.75,
    "decay_steps": 5000,
    "evaluate_step": 500
}
```
Each field in the above corresponds to the arguments of monaka.Trainer.
You can change parser model (SeqTagging to any class defined as subclass of monaka.model.LUWParserModel) and LM class (AutoLM to any class defined as subclass of monaka.module.LMEmbedding).

#### Execution
You can specify device to use and local_rank if you do distribute training.
```sh
monaka_train train --device "0" [config_file] [output_dir]
```

All checkpoint data, a log file, and configuration files are saved under [output_dir].
If you want to mointor training with tensorboad, tensorboad logs are output under [output_dir]/tb.
 
```sh
tensorboard --logdir [output_dir]/tb
```

## References and Citations
[Long Unit Word Tokenization and Bunsetsu Segmentation of Historical Japanese](https://aclanthology.org/2024.ml4al-1.6) (Ozaki et al., ML4AL-WS 2024)

```bibtex
@inproceedings{ozaki-etal-2024-long,
    title = "Long Unit Word Tokenization and Bunsetsu Segmentation of Historical {J}apanese",
    author = "Ozaki, Hiroaki  and
      Komiya, Kanako  and
      Asahara, Masayuki  and
      Ogiso, Toshinobu",
    editor = "Pavlopoulos, John  and
      Sommerschield, Thea  and
      Assael, Yannis  and
      Gordin, Shai  and
      Cho, Kyunghyun  and
      Passarotti, Marco  and
      Sprugnoli, Rachele  and
      Liu, Yudong  and
      Li, Bin  and
      Anderson, Adam",
    booktitle = "Proceedings of the 1st Workshop on Machine Learning for Ancient Languages (ML4AL 2024)",
    month = aug,
    year = "2024",
    address = "Hybrid in Bangkok, Thailand and online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.ml4al-1.6",
    doi = "10.18653/v1/2024.ml4al-1.6",
    pages = "48--55",
    abstract = "In Japanese, the natural minimal phrase of a sentence is the {``}bunsetsu{''} and it serves as a natural boundary of a sentence for native speakers rather than words, and thus grammatical analysis in Japanese linguistics commonly operates on the basis of bunsetsu units.In contrast, because Japanese does not have delimiters between words, there are two major categories of word definition, namely, Short Unit Words (SUWs) and Long Unit Words (LUWs).Though a SUW dictionary is available, LUW is not.Hence, this study focuses on providing deep learning-based (or LLM-based) bunsetsu and Long Unit Words analyzer for the Heian period (AD 794-1185) and evaluating its performances.We model the parser as transformer-based joint sequential labels model, which combine bunsetsu BI tag, LUW BI tag, and LUW Part-of-Speech (POS) tag for each SUW token.We train our models on corpora of each period including contemporary and historical Japanese.The results range from 0.976 to 0.996 in f1 value for both bunsetsu and LUW reconstruction indicating that our models achieve comparable performance with models for a contemporary Japanese corpus.Through the statistical analysis and diachronic case study, the estimation of bunsetsu could be influenced by the grammaticalization of morphemes.",
}

```