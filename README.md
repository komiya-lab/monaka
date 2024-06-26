# Monaka
A Japanese parser (including support for historical Japanese)

## Installation

## Parse

First, download  and install appropriate UniDic dictionary:
```sh
monaka download wabun
```

Available dictionaries:
| name | discription |
|------|-------------|
| gendai | 現代書き言葉 |
| spoken | 現代話し言葉 |
| novel | 近現代口語小説 |
| qkana | 旧仮名口語 |
| kindai | 近代文語 |
| kinsei | 近世江戸口語 |
| kyogen | 中世口語 |
| wakan | 中世文語 |
| wabun | 中古和文 |
| manyo | 上代語 |

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
monaka_train train --device "cuda:0" [config_file] [output_dir]
```

All checkpoint data, a log file, and configuration files are saved under [output_dir].
If you want to mointor training with tensorboad, tensorboad logs are output under [output_dir]/tb.
 
```sh
tensorboard --logdir [output_dir]/tb
```

## References and Citations
