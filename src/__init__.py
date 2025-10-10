"""
LSTM Text Autocomplete Package

Пакет для обучения и использования LSTM модели автодополнения текста.
"""

from .lstm_model import LSTMModel
from .lstm_train import train_model
from .eval_lstm import evaluate_rouge
from .next_token_dataset import NextTokenDataset, collate_fn
from .configs import CONFIG, DATA_PATH, PROJECT_PATH

__version__ = "1.0.0"
__author__ = "Yandex School of Deep Learning team"

__all__ = [
    "LSTMModel",
    "train_model", 
    "evaluate_rouge",
    "NextTokenDataset",
    "collate_fn",
    "CONFIG",
    "DATA_PATH", 
    "PROJECT_PATH"
]
