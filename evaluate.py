import hydra
import torch
import csv

from omegaconf import DictConfig
from pathlib import Path

from src.models.modules.pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer


def dict_to_device(dictornary, device):
    return {key: val.to(device) for key, val in dictornary.items()}


def evaluate(model_name, device, data, data_root):

    model = Pipeline(model_name, embedding_layer='CLS', mode='sentence').to(device)
    tokenizer = Tokenizer(model_name)

    results = {}

    print(f"Evaluating: '{model_name}'")

    return results


def to_csv(scores_dict, filename):

    should_write_header = not Path(filename).exists()

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=scores_dict.keys())
        if should_write_header:
            writer.writeheader()
        writer.writerow(scores_dict)

    print(f'Saved scores at {filename}')


@hydra.main(config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:

    data_root = cfg.data_root if cfg.data_root else Path(hydra.utils.get_original_cwd())
    output_csv = cfg.output_csv if cfg.output_csv else Path(data_root) / 'data/pruning.csv'

    results = evaluate(cfg.model_name, cfg.device, cfg.data, data_root)

    to_csv(results, output_csv)

    if Path(cfg.model_name).is_dir():
        to_csv(results, Path(cfg.model_name) / 'pruning.csv')


if __name__ == "__main__":
    main()
