# SimCLR Feature Extraction for Small Datasets

Contrastive feature learning with SimCLR on small, imbalanced image datasets using a modular PyTorch pipeline.


## Overview

This repository contains a modular PyTorch-based implementation of [SimCLR v1.0](https://arxiv.org/abs/2002.05709). It demonstrates contrastive self-supervised learning using a pretrained ResNet18 backbone.

It was developed for efficient representation learning from small, imbalanced image datasets. This is a common challenge in applied machine learning, including embedded and resource-constrained environments.

> **Note:** While the code serves as a robust proof of concept emphasizing modularity and ease of integration, it has **not** been specifically optimized or downsized for deployment on constrained hardware.

> **Note:** All images are automatically resized to **224×224 pixels**, matching the recommended input for ResNet-based models.


## Example Results

The `example_run/` folder includes a demonstration of results from a small, imbalanced 3-class dataset (~240 images). While the dataset itself is not included, it contains visualizations and training logs:

- PCA and t-SNE embeddings of learned features
- Training logs viewable via TensorBoard (`example_logs`)  

These outputs illustrate the model's ability to extract meaningful feature representations even in low-data regimes.


## Data

The dataset used during development was proprietary or task-specific, so it is **not included**.

The code is fully modular and includes built-in image resizing to **224×224 pixels**, the input size recommended by the ResNet18 backbone.

You can test the implementation easily with **ImageNet** or popular subsets like **Tiny ImageNet**.

For smaller-scale experiments, datasets like **CIFAR-10**, **STL-10**, or your own images can be used and will be resized automatically. Images must be prepared in the expected folder structure. The backbone is modular and easily exchangeable.

An example folder structure is provided in `example_data_folder/` illustrating the expected `.tar` or `.tar.gz` archive format.

See the `get_data` method in [`data_aug/contrastive_data.py`](simclr-small-dataset/data_aug/contrastive_data.py) for details on dataset integration.


## Usage

Run the full pipeline via:

```bash
python main.py
```


## Requirements

A full requirements file is provided but the lightweight summary is:

- torch
- torchvision
- matplotlib
- seaborn
- scikit-learn
- numpy
- kornia
- tqdm
- tensorboard


## Acknowledgments

Thanks to the creators of [SimCLR] (https://arxiv.org/abs/2002.05709) and the [PyTorch]community (https://pytorch.org/).
