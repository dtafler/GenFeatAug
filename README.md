# Generative Data Augmentation in the Embedding Space of Vision Foundation Models to Address Long-Tailed Learning and Privacy Constraints

## Table of Contents
- [Installation](#installation)
- [Background](#background)
- [Usage](#usage)
- [License](#license)


## Installation
1. clone the repository:
```bash
 git clone https://github.com/dtafler/GenFeatAug.git
```

2. install dependencies:
```bash
 conda env create -f ./conda_env.yaml
 conda activate cvae
```

## Background
### Abstract of Thesis
This thesis explores the potential of generative data augmentation in the embedding space of vision foundation models, aiming to address the challenges of long-tailed learning and privacy constraints. Our work leverages Conditional Variational Autoencoders (CVAEs) to enrich the representation space for underrepresented classes in highly imbalanced datasets and to enhance data privacy without compromising utility. We develop and assess methods that generate synthetic data embeddings conditioned on class labels, which both mimic the distribution of original data for privacy purposes and augment data for tail classes to balance datasets. Our methodology shows that embedding-based augmentation can effectively improve classification accuracy in long-tailed scenarios by increasing the diversity and volume of minor class samples. Additionally, we demonstrate that our approach can generate data that maintains privacy through effective anonymization of embeddings. The outcomes suggest that generative augmentation in embedding spaces of foundation models offers a promising avenue for enhancing model robustness and data security in practical applications. The findings have significant implications for deploying machine learning models in sensitive domains, where data imbalance and privacy are critical concerns.

### Overview
![Overview of the proposed methods for long-tailed classification and data
anonymization using Conditional Variational Autoencoders (CVAEs)](./background/overview_fig.png)

### Thesis
Read more [here](./background/thesis.pdf). Sections 3.3 and 3.4 detail the proposed method.

## Usage
### General
1. Create features of your image dataset with a chosen feature encoder. For example, use [this](./create_embeddings_cifar_timm.py) script to create DINOv2 features for CIFAR10 and CIFAR100, optionally also artificially imbalancing the datasets. 
2. Train a Conditional Variational Autoencoder (CVAE) on the features of your dataset. For example, use [this](./train_cvae.py) script or import the containing function to train a CVAE with various options.

#### Data Generation to improve Long-Tailed Classification
Use or adapt the Classifier class (see [here](./classifier.py)) to train a classifier on your image features plus CVAE-generated features. It utilizes the TensorDatasetTransform class (see [here](./utils/utils.py)) to generate features on the fly based on given probabilities. 

#### Data Anonymization
Use the trained CVAE to produce a entirely artificial dataset of features with any desired class-distribution, for example with the function generate_ds (see [here](./utils/utils.py)). Use the resulting dataset for any downstream task.

### Experiments
The code to run the experiments from the [thesis](./background/thesis.pdf) (Section 4), can be found in the [experiments](./experiments/) directory.

## License
