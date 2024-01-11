import json

from typing import Callable, List, Tuple
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(
        self,
        data_filename: str,
        tokenizer: Callable[[str], torch.tensor]
    ) -> None:
        
        self.data_filename = data_filename
        self.tokenizer = tokenizer

        self.target_x, self.target_y = \
            self._get_data()

    def __len__(self):
        return max(
            len(self.target_x),
            len(self.target_y),
        )

    def tokenize_and_squeeze(self, sentence: str):
        """Tokenizes a sentence and squeezes tensors (so it batchifies properly.
        """
        y = self.tokenizer(sentence)
        return {key: val.squeeze(0) for key, val in y.items()}

    def get_single_item(self, sentences: List[str], idx: int):

        length = len(sentences)
        x = sentences[idx % length]
        is_legit = (idx < length)

        y = self.tokenizer(x)
        y['is_legit'] = torch.tensor(is_legit)

        return {key: val.squeeze(0) for key, val in y.items()}

    def get_all_items(self):
        return {
            'target_x': self.tokenizer(self.target_x, return_tensors='pt'),
            'target_y': self.tokenizer(self.target_y, return_tensors='pt'),
        }

    def __getitem__(self, idx):
        return (
            self.get_single_item(self.target_x, idx),
            self.get_single_item(self.target_y, idx),
        )

    def _get_data(self) -> Tuple[List[str], List[str]]:

        with open(self.data_filename) as f:
            data = json.load(f)

        target_x = data['targ1']['examples']
        target_y = data['targ2']['examples']

        return target_x, target_y
