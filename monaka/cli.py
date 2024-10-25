import os
import sys
import typer
import json
import enum

from pathlib import Path
from typing import List, Optional
from rich.progress import Progress
from monaka.predictor import Predictor, LemmaPredictor, RESC_DIR, Encoder, Decoder
from monaka.metric import SpanBasedMetricReporter

app = typer.Typer(pretty_exceptions_show_locals=False)

class DownloadType(str, enum.Enum):
    UniDic = "unidic"

UNIDIC_URL = "https://clrd.ninjal.ac.jp/unidic_archive/"
UNIDIC_URLS = {
    "gendai": UNIDIC_URL + "2302/unidic-cwj-202302.zip",
    "spoken": UNIDIC_URL + "2302/unidic-csj-202302.zip",
    "novel": UNIDIC_URL + "2203/UniDic-202203_65_novel.zip",
    "qkana": UNIDIC_URL + "2203/UniDic-202203_60b_qkana.zip",
    "kindai": UNIDIC_URL + "2203/UniDic-202203_60a_kindai-bungo.zip",
    "kinsei": UNIDIC_URL + "2203/UniDic-202203_50c_kinsei-edo.zip",
    "kyogen": UNIDIC_URL + "2203/UniDic-202203_40_chusei-kougo.zip",
    "wakan": UNIDIC_URL + "2203/UniDic-202203_30_chusei-bungo.zip",
    "wabun": UNIDIC_URL + "2203/UniDic-202203_20_chuko.zip",
    "manyo": UNIDIC_URL + "2203/UniDic-202203_10_jodai.zip",
    "unidic-spoken": UNIDIC_URL + "2302/unidic-csj-202302.zip",
    "65_novel": UNIDIC_URL + "2203/UniDic-202203_65_novel.zip",
    "60b_qkana": UNIDIC_URL + "2203/UniDic-202203_60b_qkana.zip",
    "60a_kindai-bungo": UNIDIC_URL + "2203/UniDic-202203_60a_kindai-bungo.zip",
    "50a_kinsei-bungo": UNIDIC_URL + "2203/UniDic-202203_50a_kinsei-bungo.zip",
    "50c_kinsei-edo": UNIDIC_URL + "2203/UniDic-202203_50c_kinsei-edo.zip",
    "50b_kinsei-kamigata": UNIDIC_URL + "2203/UniDic-202203_50b_kinsei-kamigata.zip",
    "40_chusei-kougo": UNIDIC_URL + "2203/UniDic-202203_40_chusei-kougo.zip",
    "30_chusei-bungo": UNIDIC_URL + "2203/UniDic-202203_30_chusei-bungo.zip",
    "20_chuko": UNIDIC_URL + "2203/UniDic-202203_20_chuko.zip",
    "10_jodai": UNIDIC_URL + "2203/UniDic-202203_10_jodai.zip",
    "70_waka": UNIDIC_URL + "2308/unidic-waka-v202308.zip",
    "80_kansai_hougen": UNIDIC_URL + "2308/unidic-kansai-v202308.zip"
}
prv = 0

@app.command()
def download(target: str, dtype: DownloadType = typer.Option(DownloadType.UniDic, case_sensitive=False)):

    import urllib.request
    import zipfile
    import glob
    import shutil
    global prv

    if dtype == DownloadType.UniDic:
        if target not in UNIDIC_URLS:
            print(f"target: {target} is not in the UniDic dictionary name list.", file=sys.stderr)
        else:
            print(f"Downloading {target} UniDic dictionary...")
            url = UNIDIC_URLS[target]
            prv = 0
            #with typer.progressbar(length=1000, width=64, color=True) as pbar:
            with Progress() as progress:
                task = progress.add_task("[red]Downloading...", total=1000)
                def _progress(block_count: int, block_size: int, total_size: int):
                    global prv
                    size = block_size * block_count
                    val = int(size/total_size*1000)
                    progress.update(task, advance=val - prv)
                    prv = val
                fstr, msg = urllib.request.urlretrieve(url, reporthook=_progress)
            temp_path = os.path.join(RESC_DIR,".temporary")

            print(f"Extracting {target} UniDic dictionary...")
            with zipfile.ZipFile(fstr) as z:
                z.extractall(temp_path)

            target_dir = os.path.join(RESC_DIR, target)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            extracted = glob.glob(os.path.join(temp_path, "*"))
            if len(extracted) == 1: # extracted as directory
                os.rename(extracted[0], target_dir)
                os.rmdir(temp_path)
            else:
                os.rename(temp_path, target_dir)
            
            dicrc = os.path.join(target_dir, "dicrc")
            if not os.path.isfile(dicrc):
                shutil.copy(os.path.join(target_dir, ".dicrc"), dicrc)


@app.command()
def parse(model_dir: Path, inputs: List[str], device: str="cpu", batch: int=8, output_format: str="jsonl",
          tokenizer: str="mecab", dic: str="gendai", 
          node_format: str='%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[13]\t%f[27]\t%f[28]\n',
          unk_format: str='%m\t%m\t%m\t%m\tUNK\t%f[4]\t%f[5]\t\n',
          eos_format: str='EOS\n',
          bos_format: str=''
          ):
    predictor = Predictor(model_dir=model_dir)
    if len(inputs) == 1 and os.path.exists(inputs[0]):
        with open(inputs[0]) as f:
            inputs_ = [line.strip() for line in f]
    else:
        inputs_ = inputs
    for r in predictor.predict(inputs_, suw_tokenizer=tokenizer, suw_tokenizer_option={"dic": dic}, device=device, batch_size=batch, encoder_name=output_format, node_format=node_format, unk_format=unk_format, eos_format=eos_format, bos_format=bos_format):
        print(r)

@app.command()
def predict(model_dir: Path, input_file: Path, device: str="cpu", batch: int=8, output_format: str="jsonl",
          tokenizer: str="mecab", dic: str="gendai", 
          node_format: str='%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[13]\t%f[27]\t%f[28]\n',
          unk_format: str='%m\t%m\t%m\t%m\tUNK\t%f[4]\t%f[5]\t\n',
          eos_format: str='EOS\n',
          bos_format: str=''
          ):
    with open(input_file) as f:
        inputs = [json.loads(line)["sentence"] for line in f]
    predictor = Predictor(model_dir=model_dir)
    for r in predictor.predict(inputs, suw_tokenizer=tokenizer, suw_tokenizer_option={"dic": dic}, device=device, batch_size=batch, encoder_name=output_format, node_format=node_format, unk_format=unk_format, eos_format=eos_format, bos_format=bos_format):
        print(r)


@app.command()
def predict_lemma(model_dir: Path, input: str):
    predictor = LemmaPredictor(model_dir=model_dir)
    print(predictor.predict(input))

@app.command()
def evaluate_lemma(model_dir, inputfile: str):
    predictor = LemmaPredictor(model_dir=model_dir)
    print(json.dumps(predictor.evaluate(inputfile), indent=True, ensure_ascii=False))


@app.command()
def evaluate(model_dir, inputfile: str, device: str="cpu", batch: int=8, targets: List[str]=("luw", "chunk"), pos_level:int = -1, format: str="pretty", outputfile: str=None):
    predictor = Predictor(model_dir=model_dir)
    predictor.evaluate(inputfile, batch_size=batch, device=device, targets=targets, pos_level=pos_level, format_=format, outputfile=outputfile)


@app.command()
def score(gold_file: Path, pred_file: Path, targets: List[str]=("suw_span", "luw_span", "luw_triple", "chunk_span"), format: str="pretty"):

    def to_tuple(data, L=2):
        return [tuple(d[:L]) for d in data]

    reporters = {t:SpanBasedMetricReporter(t) for t in targets}
    with open(gold_file) as gf, open(pred_file) as pf:
        for gline, pline in zip(gf, pf):
            gjs = json.loads(gline)
            pjs = json.loads(pline)
            for t in targets:
                rep = reporters[t]
                if "span" in t:
                    rep.update(to_tuple(gjs[t]), to_tuple(pjs[t]))
                else:
                    target = t.replace("triple", "span")
                    rep.update(to_tuple(gjs[target], 3), to_tuple(pjs[target], 3))


    for rep in reporters.values():
        if format == "pretty":
            rep.pretty()
        else:
            print(json.dumps(rep.to_json(), indent=True))



@app.command()
def convert(dencoder: str, encoder: str, file_path: Path):
    enc = Encoder.by_name(encoder)()
    dec = Decoder.by_name(dencoder)()
    with open(file_path) as f:
        for line in f:
            js = json.loads(line)
            data = dec.decode(**js)
            out = enc.encode(**data)
            print(out)


if __name__ == "__main__":
    app()
