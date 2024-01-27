import os
import typer
import json

from pathlib import Path
from typing import List
from monaka.trainer import Trainer

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def create_vocab(jsonl_files: List[Path], output_dir: Path):
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
def train(config_file: str, output_dir: str, device: str="cpu", local_rank: int=-1):
    with open(config_file) as f:
        config = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=True, ensure_ascii=False)

    trainer = Trainer(output_dir=output_dir, **config)
    trainer.train(device, local_rank)


if __name__ == "__main__":
    app()

    