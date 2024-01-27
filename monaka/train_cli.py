import os
import typer
import json
import numpy as np

from pathlib import Path
from typing import List, Optional
from monaka.trainer import Trainer

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def create_vocab(output_dir: Path, jsonl_files: List[Path]):
    os.makedirs(output_dir, exist_ok=True)
    pos = {"unk": 0, "pad": 1}
    labels = {"unk": 0, "pad": 1}
    for jsonl_file in jsonl_files:
        with open(jsonl_file) as f:
            for line in f:
                js = json.loads(line)
                for p in js["pos"]:
                    if p not in pos:
                        pos[p] = len(pos)
                for l in js["labels"]:
                    if l not in labels:
                        labels[l] = len(labels)
    
    with open(os.path.join(output_dir, "pos.json"), "w") as f:
        json.dump(pos, f, indent=True, ensure_ascii=False)

    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=True, ensure_ascii=False)


@app.command()
def train(config_file: str, output_dir: str, device: str="cpu", local_rank: int=-1, 
          train_files: Optional[List[Path]]=None, dev_files: Optional[List[Path]]=None, test_files: Optional[List[Path]]=None):
    with open(config_file) as f:
        config = json.load(f)

    if train_files is not None:
        config["train_files"] = train_files

    if dev_files is not None:
        config["dev_files"] = dev_files

    if test_files is not None:
        config["test_files"] = test_files

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=True, ensure_ascii=False)

    trainer = Trainer(output_dir=output_dir, **config)
    trainer.train(device, local_rank)


@app.command()
def create_split(output_dir: Path, jsonl_files: List[Path], dev_ratio: float=0.05, test_ratio: float=0.05, folds: int=5):
    os.makedirs(output_dir, exist_ok=True)
    used = list()
    
    for k in range(folds):
        print(f"createing fold num.:{k}")
        trains = list()
        devs = list()
        tests = list()

        for fname in jsonl_files:
                with open(fname) as f:
                    lines = [line for line in f if line]
                n_lines = [l for l in lines if l not in used]
                u_lines = [l for l in lines if l in used]

                np.random.shuffle(n_lines)
                dend = int(len(lines) * dev_ratio)
                tend = dend + int(len(lines) * test_ratio)
                devs.extend(n_lines[:dend])
                tests.extend(n_lines[dend:tend])
                trains.extend(n_lines[tend:])
                trains.extend(u_lines)

        used.extend(tests)


        with open(os.path.join(output_dir, f"train.{k}.jsonl"), "w") as f:
            for line in trains:
                f.write(line)
        
        with open(os.path.join(output_dir, f"dev.{k}.jsonl"), "w") as f:
            for line in devs:
                f.write(line)

        with open(os.path.join(output_dir, f"test.{k}.jsonl"), "w") as f:
            for line in tests:
                f.write(line)

if __name__ == "__main__":
    app()

    