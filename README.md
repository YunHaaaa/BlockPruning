
## Reproducibility
Setup
```bash
conda env create -f envs/pruner.yaml
conda activate pruning
pip uninstall nn_pruning
pip install git+https://github.com/[anonymized]/nn_pruning.git@automodel
```

Block pruning
```bash
python run.py --multirun \
    experiment=block_pruning_frozen \
    model.embedding_layer=last,all \
    model.mode=sentence,token \
    prune_block_size=32,64
```

Pruning enitre heads
```bash
python run.py --multirun \
    experiment=head_pruning_frozen_values_only \
    model.embedding_layer=last,all \
    model.mode=sentence,token
```

only:
```bash
python run.py --multirun \
    model.embedding_layer=first,last,all,intermediate \
    model.mode=sentence,token
```


## Credits
* Block pruning:
```bibtex
@article{Lagunas2021BlockPF,
  title={Block Pruning For Faster Transformers},
  author={Franccois Lagunas and Ella Charlaix and Victor Sanh and Alexander M. Rush},
  journal={ArXiv},
  year={2021},
  volume={abs/2109.04838}
}
```
* The original debiaing idea:
```bibtex
@inproceedings{kaneko-bollegala-2021-context,
    title={Debiasing Pre-trained Contextualised Embeddings},
    author={Masahiro Kaneko and Danushka Bollegala},
    booktitle = {Proc. of the 16th European Chapter of the Association for Computational Linguistics (EACL)},
    year={2021}
}
```
* Hydra+lightning template by [ashleve](https://github.com/ashleve/lightning-hydra-template).
