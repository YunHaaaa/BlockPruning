from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule


from src.utils.utils import get_logger
from src.models.modules.pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer


from transformers import AdamW, get_linear_schedule_with_warmup

log = get_logger(__name__)


@dataclass(unsafe_hash=True)
class Debiaser(LightningModule):

    model_name: str
    embedding_layer: str
    debias_mode: str
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

        self.model_debias = Pipeline(
            model_name=self.model_name,
            embedding_layer=self.embedding_layer,
            debias_mode=self.debias_mode,
            hf_checkpoint=self.hf_checkpoint,
            is_glue=self.is_glue
        )
        self.model_original = Pipeline(
            model_name=self.model_name,
            embedding_layer='all',   # See Eq. (3)
            debias_mode='sentence',  # See Eq. (3)
            hf_checkpoint=self.hf_checkpoint,
            is_glue=self.is_glue
        )

        self.tokenizer = Tokenizer(self.model_name)

    def forward(self, inputs, return_word_embs=None, embedding_layer=None):
        """Forward pass of the model to be debiased."""
        return self.model_debias(inputs, return_word_embs, embedding_layer)

    def forward_original(self, inputs, return_word_embs=None, embedding_layer=None):
        """Forward pass of the original model (frozen)."""
        with torch.no_grad():
            return self.model_original(inputs, return_word_embs, embedding_layer)

    def loss_regularize(self, attributes, attributes_original):
        """Loss for regularization (L2), Eq.(3)

        Args: contextualied embeddings of attributes, wrt to debiased
            and original model, respectively. Both are of shape:
            (batch_sz * n, emb_dim), where
            n = num_layers if embedding_layer=='all' else 1.
        """
        assert attributes.shape == attributes_original.shape
        return (attributes - attributes_original).pow(2).sum(1).mean()

    def step(self, batch) -> Dict[str, float]:
        """A step performed on training and validation.

        This is basically Eq.(4) in the paper.

        It computes debiasing loss with the regularizer term.

        Note, that in the regularization term, *word* embeddings are taken
        across *all* layers in both models (see Eq. 3).
        """
        loss = self.loss_alpha  + self.loss_beta 

        return loss

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
            self.model_debias.parameters(),
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
