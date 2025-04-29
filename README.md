# Ensemble Inference for LLMs
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jniimi/ensemble_inference/blob/main/sample.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2504.18884-b31b1b.svg)](https://arxiv.org/abs/2504.18884)

Niimi, J. (2025) "A Simple Ensemble Strategy for LLM Inference: Towards More Stable Text Classification" In Proceedings of the 30th International Conference on Natural Language & Information Systems (NLDB 2025)

## Overview
A simple method of ensemble to aggregate multiple inferences from different LLMs to improve the robustness of the inference. Proposed by Niimi (2025) in NLDB2025.

## Requirements
- pandas>=2.0.1
- transformers>=4.51.3
- bitsandbytes>=0.45.5
- torch>=2.6.0

## Installation
You can load the scripts with `git clone` and incorporate into your analyses. 
```
git clone https://github.com/jniimi/ensemble_inference
cd ensemble_inference
pip install -r requirements.txt
import ensemble_inference as ens
```
This approach can be implemented in any LLMs; however, the models with wide pretraining and instruction-tuning are highly recommended. This example adopts `Llama-3-8B-Instruct`.

### You can refer sample on Google Colab
[https://colab.research.google.com/github/jniimi/ensemble_inference/blob/main/sample.ipynb](https://colab.research.google.com/github/jniimi/ensemble_inference/blob/main/sample.ipynb)

## Reference
```
@inproceedings{niimi2025nldb,
  author = {Junichiro Niimi},
  title = {A Simple Ensemble Strategy for LLM Inference: Towards More Stable Text Classification},
  booktitle = {Proceedings of the 30th International Conference on Natural Language & Information Systems (NLDB 2025)},
  doi = {10.48550/arXiv.2504.18884},
  year = {2025},
  publisher = {Springer}
}
```
