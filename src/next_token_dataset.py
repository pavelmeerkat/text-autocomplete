import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict


class NextTokenDataset(Dataset):
    def __init__(self, texts: List[List[str]], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    def get_length_group(item: Dict[str, torch.Tensor]) -> int:
        length = len(item["input_ids"])
        if length <= 10:
            return 0
        elif length <= 20:
            return 1
        elif length <= 40:
            return 2
        else:
            return 3
    
    batch.sort(key=lambda x: (get_length_group(x), len(x["input_ids"])), reverse=True)
    
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded
    }