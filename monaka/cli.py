import os
import sys
import typer
import json
import enum

from pathlib import Path
from typing import List, Optional
from monaka.predictor import Predictor, RESC_DIR

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
    "manyo": UNIDIC_URL + "2203/UniDic-202203_10_jodai.zip"
}
"""
pbar: Optional[typer.progressbar] = None

def _progress(block_count: int, block_size: int, total_size: int):
    global pbar
    if pbar is None:
        pbar = typer.progressbar(length=total_size)#tqdm(total=total_size)
    else:
        size = block_size * block_count
        pbar.update(size)
        # reset global bar
        #if pbar.
        #    pbar = None
"""
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
            with typer.progressbar(length=1000, width=64, color=True) as pbar:
                def _progress(block_count: int, block_size: int, total_size: int):
                    global prv
                    size = block_size * block_count
                    val = int(size/total_size*1000)
                    pbar.update(val - prv)
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
          tokenizer: str="mecab", dic: str="gendai"):
    predictor = Predictor(model_dir=model_dir)
    for r in predictor.predict(inputs, suw_tokenizer=tokenizer, suw_tokenizer_option={"dic": dic}, device=device, batch_size=batch, encoder_name=output_format):
        print(r)
    

@app.command()
def evaluate(model_dir, inputfile: str, device: str="cpu", batch: int=8, targets: List[str]=("luw", "chunk")):
    predictor = Predictor(model_dir=model_dir)
    predictor.evaluate(inputfile, batch_size=batch, device=device, targets=targets)

if __name__ == "__main__":
    app()
