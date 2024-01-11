from dataclasses import dataclass
from typing import Dict
from pathlib import Path
from pytorch_lightning import LightningDataModule
from datasets import load_from_disk

from torch.utils.data import DataLoader
from torchtext.utils import download_from_url, extract_archive

from src.models.modules.tokenizer import Tokenizer
from src.dataset.utils import extract_data
from src.utils.utils import get_logger


log = get_logger(__name__)


@dataclass(unsafe_hash=True)
class DebiasDataModule(LightningDataModule):
    model_name: str
    batch_size: int
    data_dir: str
    datafiles: Dict[str, str]
    seed: int
    num_proc: int      # For dataset preprocessing
    num_workers: int   # For dataloaders


    def __post_init__(self):
        super().__init__()
        self.tokenizer = Tokenizer(self.model_name)
        self.data_dir = Path(self.data_dir)

        self.dataset_cache = self.data_dir / "dataset" / self.model_name / str(self.seed)


    def prepare_data(self):
        datafiles = {}
        for name, url in self.datafiles.items():
            download_path = download_from_url(url, root=self.data_dir)
            datafiles[name] = download_path
            if download_path.endswith('.gz'):
                extracted_path = extract_archive(download_path, self.data_dir)[0]
                datafiles[name] = extracted_path

        # The first call will cache the data
        if not self.dataset_cache.exists():
            log.info(f"Processing and caching the dataset to {self.dataset_cache}.")
            extract_data(
                rawdata_path=datafiles['plaintext'],
                model_name=self.model_name,
                data_root=self.dataset_cache,
                num_proc=self.num_proc
            )
        else:
            log.info(f"Reading cached datset at {self.dataset_cache}")


    def setup(self):
        # Data is cached to disk now
        data = load_from_disk(self.dataset_cache)
        data.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'keyword_mask']
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
