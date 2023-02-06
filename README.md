# KG-TORE: Tailored recommendations through knowledge-aware GNN models

This is the official implementation of the paper
*KG-TORE: Tailored recommendations through knowledge-aware GNN models* submitted to *SIGIR Conference on Research and Development in Information Retrieval*.

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
  - [Installation guidelines: scenario #1](#installation-guidelines-scenario-1)
  - [Installation guidelines: scenario #2](#installation-guidelines-scenario-2)
- [Datasets](#datasets)
- [Elliot Configuration Files](#elliot-configuration-files)
- [Recommendation Lists](#recommendation-lists)
- [KGTORe Parameters](#kgtore-parameters)
- [Usage](#usage)
  - [Reproduce Paper Results](#reproduce-paper-results)
  - [Preprocessing](#preprocessing)
  - [Evaluate Recommendation Lists](#evaluate-recommendation-lists)
  - [Statistical Significance](#statistical-significance)



## Description

The code in this repository allows replicating the experimental setting described within the paper.

The recommenders training and evaluation procedures have been developed on the reproducibility framework **Elliot**,
so we suggest you refer to the official GitHub 
[page](https://github.com/sisinflab/elliot) and 
[documentation](https://elliot.readthedocs.io/en/latest/).

Regarding the graph-based recommendation models based on torch, they have been implemented
in `PyTorch Geometric` using the version `1.10.2`, with CUDA `10.2` and cuDNN `8.0`

For granting the usage of the same environment on different machines, 
all the experiments have been executed on the same docker container.
If the reader would like to use it, 
please look at the corresponding section in [requirements](#requirements).

## Requirements 

This software has been executed on the operative system Ubuntu `18.04`.

Please, make sure to have the following installed on your system:

* Python `3.8.0` or later
* PyTorch Geometric with PyTorch `1.10.2` or later
* CUDA `10.2`

### Installation guidelines: scenario #1
If you have the possibility to install CUDA on your workstation (i.e., `10.2`), you may create the virtual environment with the requirements files we included in the repository, as follows:

```
# PYTORCH ENVIRONMENT (CUDA 10.2, cuDNN 8.0)

$ python3.8 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
```

### Installation guidelines: scenario #2
A more convenient way of running experiments is to instantiate a docker container having CUDA `10.2` already installed.

Make sure you have Docker and NVIDIA Container Toolkit installed on your machine (you may refer to this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)).

Then, you may use the following Docker image to instantiate the container equipped with CUDA `10.2` and cuDNN `8.0` (the environment for `PyTorch`): [link](https://hub.docker.com/layers/nvidia/cuda/10.2-cudnn8-devel-ubuntu18.04/images/sha256-3d1aefa978b106e8cbe50743bba8c4ddadacf13fe3165dd67a35e4d904f3aabe?context=explore)

After the setup of your Docker containers, you may follow the exact same guidelines as [scenario #1](#installation-guidelines-scenario-1).

## Datasets

At `./data/` you may find all the [files](data) related to 
the datasets, the knowledge graphs and the related item-entity linking.

The datasets could be found within the directory `./data/[DATASET]/data`. 
Only for Movielens 1M, within the [directory](data/movielens/grouplens) `./data/movielens/grouplens`
For the knowledge graphs and links please look at  `./data/[DATASET]/dbpedia`.

## Elliot Configuration Files

At `./config_files/` you may find the Elliot [configuration files](config_files) used for setting the experiments.


The configuration files for training the models are reported as `[DATASET]_[MODEL].yml`. 
While the best models hyperparameters are reported in the files named `[DATASET]_best_[MODEL].yml`. 

## Recommendation Lists

The best models recommendation lists could be found at `./results/recs`. 
You may be use them for computing the recommendation metrics as described at [Evaluate Recommendation](#Evaluate-Recommendation-Lists)

### KGTORe Parameters

The following are the parameters required by KGTORe:
- ```batch size```: training batch size;
- ```lr```: learning rate;
- ```elr```: features embedding learning rate;
- ```l_w```: embedding regularization;
- ```l_ind```: independence loss weight;
- ```alpha```: alpha parameter;
- ```beta```: beta parameter;
- ```factors```: embeddings dimension;
- ```ind_edges```: fraction of edges selected for computing the independence loss in each training batch;
- ```n_layers```: graph convolutional network layers;
- ```npr```: negative-positive ratio when building the decision tree;
- ```epochs```: training epochs

## Usage

Here we describe the steps to reproduce the results presented in the paper. 
Furthermore, we provide a description of how the experiments have been configured.

### Reproduce Paper Results

[Here](run.py) you can find a ready-to-run Python file with all the pre-configured experiments cited in our paper.
You can easily run them with the following command:

```
python run.py
```

It runs the pre-processing procedure and then trains our KGTORe model and all the baselines on three different datasets.
The results will be stored in the folder ```results/DATASET/```.

### Preprocessing

If you are interested in running just the data preprocessing step, please run:

```
python preprocessing.py
```

### Evaluate Recommendation Lists

For computing the recommendation metrics on the recommendation lists run the following command:

```
python compute_metrics_on_recs.py
```

Within the [file](compute_metrics_on_recs.py) it is possible to specify the directory containing the recommendation lists.
The [default](results/recs/) solution is ```results/recs/[DATASET]```.

Along with the evaluation metrics also the student's paired t-test is computed.

### Statistical Significance

The results statistical significance has been evaluated performing a Student's Paired t-Test. 
The precomputed results on the best models could be found at
[results/student_paired_t_test](results/student_paired_t_test).

The paired t-test is computed during recommendation metrics computation. 
Please, see the [evaluation](#evaluate-recommendation-lists) section to compute them.
