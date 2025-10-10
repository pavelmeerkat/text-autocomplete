from pathlib import Path

import evaluate
import pandas as pd
import yaml
from yaml.loader import SafeLoader
from transformers import AutoTokenizer


PROJECT_PATH = Path(__file__).parent.parent
DATA_PATH = PROJECT_PATH / 'data'


yaml_path = PROJECT_PATH / 'configs' / 'config.yaml'
with open(yaml_path) as file:
    CONFIG = yaml.load(file, Loader=SafeLoader)


ROUGE = evaluate.load("rouge")

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")