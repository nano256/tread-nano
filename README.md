<h2 align="center">🧹CleanDIFT: Diffusion Features without Noise</h2>
<div align="center"> 
  <a href="" target="_blank">Felix Krause</a> · 
  <a href="" target="_blank">Timy Phan</a> · 
  <a href="" target="_blank">Vincent Tao Hu</a> · 
  <a href="https://ommer-lab.com/people/ommer/" target="_blank">Björn Ommer</a>
</div>
<p align="center"> 
  <b>CompVis Group @ LMU Munich</b> <br/>
</p>

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2501.04765)

This repository contains the official implementation of the paper "TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training".


We propose TREAD, a new method to increase the efficiency of diffusion trainig by increasing both iteration speed and performance at the same time. For this we use uni-directional token transportation to module the information flow in the network. 

<div align="center">
  <img src="./docs/images/teaser.png" alt="teaser" style="width:50%;">
</div>

## 🚀 Usage

### Training

In order to train a diffusion model, we offer a minimalistic training script in `train.py`. In its simplest form, it can be started using:
```python
accelerate launch train.py
```
with `configs.config.yaml` haveing all relevant information and settings for the actual training run. 
`Note:` We expect precomputed latents in this version. 
Under `model` one can decide between `dit`and `tread` which are the preconfigured version here with the former being the standard dit and the latter being supported by TREAD. How these changes are implemented can be seen in `dit.py` and `routing_module.py`. 

### Sampling

For sampling we use the [EDM](https://github.com/NVlabs/edm) sampling  and the FID calculation is done via the [ADM](https://github.com/openai/guided-diffusion) evaluation suite . We provide a `fid.py` to evaluate our models during training with the same reference batches as ADM.

## Acknowledgements
Thanks to the open source codebases such as [DiT](https://github.com/facebookresearch/DiT), [MaskDiT](https://github.com/Anima-Lab/MaskDiT), [ADM](https://github.com/openai/guided-diffusion), and [EDM](https://github.com/NVlabs/edm). Our codebase is built on them.
