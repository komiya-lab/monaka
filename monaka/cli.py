import os
import typer
import json

from pathlib import Path
from typing import List
from monaka.trainer import Trainer

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def parse():
    print("hello")


if __name__ == "__main__":
    app()
