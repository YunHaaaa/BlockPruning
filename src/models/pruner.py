from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule


from src.utils.utils import get_logger
from src.models.modules.pipeline import Pipeline

from transformers import AdamW, get_linear_schedule_with_warmup

log = get_logger(__name__)


@dataclass(unsafe_hash=True)
class Pruner(LightningModule):

    model_name: str
    embedding_layer: str
    mode: str
    learning_rate: float
    weight_decay: float
    adam_eps: float
    warmup_steps: int
    loss_alpha: float
    loss_beta: float
    hf_checkpoint: str = None
    is_glue: bool = False

    # Used by child only
    sparse_train_args: Dict[str, Any] = None
    freeze_weights: bool = False
    share_pruning_scores: bool = False
    prune_values_only: bool = False
    prune_attention_only: bool = False

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model_pruning = Pipeline(
            model_name=self.model_name,
            embedding_layer=self.embedding_layer,
            mode=self.mode,
            hf_checkpoint=self.hf_checkpoint,
            is_glue=self.is_glue
        )
        self.model_original = Pipeline(
            model_name=self.model_name,
            embedding_layer='all',   # See Eq. (3)
            mode='sentence',  # See Eq. (3)
            hf_checkpoint=self.hf_checkpoint,
            is_glue=self.is_glue
        )

    def forward(self, inputs, return_word_embs=None, embedding_layer=None):
        return self.model_pruning(inputs, return_word_embs, embedding_layer)

    def forward_original(self, inputs, return_word_embs=None, embedding_layer=None):
        """Forward pass of the original model (frozen)."""
        with torch.no_grad():
            return self.model_original(inputs, return_word_embs, embedding_layer)

    def loss_regularize(self, attributes, attributes_original):
        assert attributes.shape == attributes_original.shape
        return (attributes - attributes_original).pow(2).sum(1).mean()

    def step(self, batch) -> Dict[str, float]:
        loss = self.loss_alpha  + self.loss_beta 

        return loss

    # def step(self, batch) -> Dict[str, float]:
    #     targets = self(batch["targets"])

    #     attributes = self(
    #         batch['attributes'], return_word_embs=True, embedding_layer='all'
    #     )
    #     attributes_original = self.forward_original(
    #         batch['attributes'], return_word_embs=True, embedding_layer='all'
    #     )

    #     loss_debias = self.loss_debias(
    #         static_attributes=self.non_contextualized, targets=targets
    #     )
    #     loss_regularize = self.loss_regularize(
    #         attributes=attributes, attributes_original=attributes_original
    #     )

    #     loss = self.loss_alpha * loss_debias + self.loss_beta * loss_regularize

    #     return {
    #         "loss": loss,
    #         "loss_debias": loss_debias,
    #         "loss_regularize": loss_regularize
    #     }


    def log_loss(self, loss:  float, stage: str):
        self.log(
            f"{stage}/loss", loss,
            prog_bar=False, on_epoch=True, sync_dist=True
        )

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log_loss(loss, 'train')
        return loss["loss"]

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log_loss(loss, 'validation')

    @property
    def total_train_steps(self):
        num_devices = 1
        if self.trainer.gpus and self.trainer.gpus > 0:
            if isinstance(self.trainer.gpus, list):
                num_devices = len(self.trainer.gpus)
            else:
                num_devices = self.trainer.gpus

        # Be carefull: trainloader is a dict of loaders of equal length
        num_samples = len(self.train_dataloader()["targets"])
        train_batches = num_samples // num_devices
        total_epochs = self.trainer.max_epochs - self.trainer.min_epochs + 1

        return (total_epochs * train_batches) // self.trainer.accumulate_grad_batches

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model_pruning.parameters(),
            weight_decay=self.weight_decay,
            lr=self.learning_rate,
            eps=self.adam_eps
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_train_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
