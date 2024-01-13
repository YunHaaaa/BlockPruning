from typing import Dict, List
from pathlib import Path
from datasets import load_dataset
from src.models.modules.tokenizer import Tokenizer
from datasets.utils import disable_progress_bar

import regex as re
import torch

disable_progress_bar()


def get_keyword_set(filepath: str) -> set:
    """Reads file with keywords and returns a set containing them all"""

    # This is cumbersome: hydra creates own build dir,
    # where data/ is not present. We need to escape to original cwd
    import hydra  # TODO(Przemek): do it better. Maybe download data?
    quickfix_path = Path(hydra.utils.get_original_cwd()) / filepath

    with open(quickfix_path) as f:
        return {line.strip() for line in f.readlines()}


def extract_data(
    rawdata_path: Path,
    model_name: str,
    data_root: Path,
    num_proc: int,
) :
    tokenizer = Tokenizer(model_name)

    # This regexp basically tokenizes a sentence over spaces and 's, 're, 've..
    # It's originally taken from OpenAI's GPT-2 Encoder implementation
    pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")  # noqa E501

    np = 64
    # https://huggingface.co/docs/datasets/process.html#save

    dataset = load_dataset('text', data_files=rawdata_path, split="train[:]")
    dataset = dataset.map(lambda examples: tokenizer(examples['text']), num_proc=np)

    dataset = dataset.train_test_split(test_size=1000)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'keyword_mask'])

    dataset.save_to_disk(data_root / "dataset")

    return dataset

