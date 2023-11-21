# Kilter DL

This repository has various half-finished models and scripts for dealing with KilterBoard climbs (generating/grading/autoencoding).

## Installation

```python
conda create -n kilter -f env.yaml
```

## Preprocessing

Use the `preprocess.ipynb` notebook to preprocess the data, generating the necessary csv files in the `data/` directory.

## Diffusion model

Use the `python diffuse.py` command to train a diffusion model for climb generation.

## Prediction ViT model

Use the `python train.py` command to train a model to predict the difficulty of the climb.

## Sampling

Use the `sample.ipynb` notebook to load trained diffusion and prediction models together to generate climbs.
