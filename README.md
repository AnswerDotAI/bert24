# Welcome!

This is the repository where you can find ModernBERT, our experiments to bring BERT into modernity via both architecture changes and scaling.

This repository noticeably introduces FlexBERT, our modular approach to encoder building blocks, and heavily relies on .yaml configuration files to build models. The codebase builds upon [MosaicBERT](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert), and specifically the [unmerged fork bringing Flash Attention 2](https://github.com/Skylion007/mosaicml-examples/tree/skylion007/add-fa2-to-bert) to it, under the terms of its Apache 2.0 license. We extend our thanks to MosaicML for starting the work on modernising encoders! 

This README is very barebones and is still under construction. It will improve with more reproducibility and documentation in the new year, as we gear up for more encoder niceties after the pre-holidays release of ModernBERT. For now, we're mostly looking forward to seeing what people build with the [🤗 model checkpoints](https://huggingface.co/collections/answerdotai/modernbert-67627ad707a4acbf33c41deb)).

All code used in this repository is the code used as part of our experiments for both pre-training and GLUE evaluations, there's no uncommitted secret training sauce.

** This is the research repository for ModernBERT, focused on pre-training and evaluations. If you're seeking the HuggingFace version, designed to integrate with any common pipeline, please head to the [ModernBERT Collection on HuggingFace](https://huggingface.co/collections/answerdotai/modernbert-67627ad707a4acbf33c41deb)**

## Setup

We have fully documented the environment used to train ModernBERT, which can be installed on a GPU-equipped machine with the following commands:

```bash
conda env create -f environment.yaml
# if the conda environment errors out set channel priority to flexible:
# conda config --set channel_priority flexible
conda activate bert24
# if using H100s clone and build flash attention 3
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention/hopper
# python setup.py install
# install flash attention 2 (model uses FA3+FA2 or just FA2 if FA3 isn't supported)
pip install "flash_attn==2.6.3" --no-build-isolation
# or download a precompiled wheel from https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3
# or limit the number of parallel compilation jobs
# MAX_JOBS=8 pip install "flash_attn==2.6.3" --no-build-isolation
```

## Training

Training heavily leverages the [composer](https://github.com/mosaicml/composer) framework. All training are configured via YAML files, of which you can find examples in the `yamls` folder. We highly encourage you to check out one of the example yamls, such as `yamls/main/flex-bert-rope-base.yaml`, to explore the configuration options.


## Evaluations

### GLUE

GLUE evaluations for a ModernBERT model trained with this repository can be ran with via `run_evals.py`, by providing it with a checkpoint and a training config. To evaluate non-ModernBERT models, you should use `glue.py` in conjunction with a slightly different training YAML, of which you can find examples in the `yamls/finetuning` folder.

### Retrieval

The `examples` subfolder contains scripts for training retrieval models, both dense models based on [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) and ColBERT models via the [PyLate](https://github.com/lightonai/pylate) library:
- `examples/train_pylate.py`: The boilerplate code to train a ModernBERT-based ColBERT model with PyLate.
- `examples/train_st.py`: The boilerplate code to train a ModernBERT-based dense retrieval model with Sentence Transformers.
- `examples/evaluate_pylate.py`: The boilerplate code to evaluate a ModernBERT-based ColBERT model with PyLate.
- `examples/evaluate_st.py`: The boilerplate code to evaluate a ModernBERT-based dense retrieval model with Sentence Transformers.


## Reference

If you use ModernBERT in your work, be it the released models, the intermediate checkpoints (release pending) or this training repository, please cite:

```bibtex
@misc{modernbert,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference}, 
      author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
      year={2024},
      eprint={2412.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13663}, 
}
```