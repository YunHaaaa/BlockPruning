from dataclasses import dataclass
from datasets import load_dataset
from pathlib import Path
from pytorch_lightning import LightningDataModule
from datasets import load_from_disk

from torch.utils.data import DataLoader

from src.utils.utils import get_logger


log = get_logger(__name__)


@dataclass(unsafe_hash=True)
class DataModule(LightningDataModule):
    model_name: str
    batch_size: int
    data_dir: str
    dataset_name: str
    task_name: str
    train_file: str
    validation_file: str
    test_file: str
    seed: int
    num_proc: int      # For dataset preprocessing
    num_workers: int   # For dataloaders


    def __post_init__(self):
        super().__init__()
        self.data_dir = Path(self.data_dir)

        self.dataset_cache = self.data_dir / "dataset" / self.model_name / str(self.seed)

    def prepare_data(self):

        if self.task_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset("glue", self.data_args.task_name, cache_dir=self.dataset_cache)
        elif self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.data_args.dataset_name, self.data_args.dataset_config_name, cache_dir=self.dataset_cache
            )
        else:
            # Loading a dataset from your local files.
            data_files = {"train": self.data_args.train_file, "validation": self.data_args.validation_file}

            if self.training_args.do_predict:
                if self.data_args.test_file is not None:
                    train_extension = self.data_args.train_file.split(".")[-1]
                    test_extension = self.data_args.test_file.split(".")[-1]
                    assert (
                        test_extension == train_extension
                    ), "`test_file` should have the same extension (csv or json) as `train_file`."
                    data_files["test"] = self.data_args.test_file
                else:
                    raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

            for key in data_files.keys():
                log.info(f"load a local file for {key}: {data_files[key]}")

            if self.data_args.train_file.endswith(".csv"):
                raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=self.dataset_cache)
            else:
                raw_datasets = load_dataset("json", data_files=data_files, cache_dir=self.dataset_cache)

        return raw_datasets


    def setup(self):
        # Data is cached to disk now
        data = load_from_disk(self.dataset_cache)
        data.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask']
        )
        # Targets (stereotypes)
        self.train = data['train']
        self.val = data['test']


    def train_dataloader(self):
        assert self.trainer.reload_dataloaders_every_n_epochs == 1

        train = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train
    

    def val_dataloader(self):

        val = DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val
