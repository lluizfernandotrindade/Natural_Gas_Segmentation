# This code allows to reproduce results of our article:
## Improving Generalization through Self-Supervised Learning Using Generative Pretraining Transformer for Natural Gas Segmentation 

This project objective is to train an Deep Learning model with the post stack seismic 2D/3D data for natural gas detection.

It can process multiple pipelines and for data loading and data handling to create samples for training and inference

## Environment Setup

### Prerequisites: 

* Python 3.11 
* CUDA Toolkit 11.8 
* CUDA Deep Neural Network (cuDNN) 8.8 
* CUDA Compiler Driver - NVCC 11.8 (When missing drivers)

### Installation via Anaconda or Miniconda

Before starting consider having Anaconda or Miniconda installed. To prepare the environment, follow these steps:

#### 1. Create an environment with a name of your choice:

```bash
conda create -n {my_env_name} python=3.11
conda activate {my_env_name}
```

> In case of shared environments, consider using the path to your environment instead of a global name:

```bash
conda create -p {path/to/my/env} python=3.11
conda activate {path/to/my/env}
```

#### 2. Install the necessary packages and drivers:

```bash
conda env update -n {my_env_name} -f environment.yml
# conda env update -p {path/to/my/env} -f environment.yml
```

Or install manually

```bash
conda install -c conda-forge cudatoolkit=11.8
conda install -c conda-forge cudnn=8.8
conda install -c nvidia cuda-nvcc=11.8
```

#### 3. Update the environment variables in your configuration:

```bash
conda env config vars set LD_LIBRARY_PATH=/usr/lib:/usr/lib64:/var/lib:${CONDA_PREFIX}/lib

# When CUDA Compiler Driver is missing
conda env config vars set XLA_FLAGS=--xla_gpu_cuda_data_dir=${CONDA_PREFIX}
```

#### 4. Install the remaining packages via pip:

```bash
pip install -r requirements.txt
```

### Manual Installation

You can choose to install the drivers manually, but using multiple versions of drivers can compromise your working environment. Do so at your own risk.

## Project Structure

```
├── README.md          <- The top-level README for developers using this project.
│
├── outputs            <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks.
│
├── config.yaml        <- The default application execution behavior.
│
├── requirements.txt   <- The requirements file for create the environment.
│
├── environment.yml    <- The requirements file for create the anaconda environment.
│
└── src/   <- Source code for use in this project.
    │
    ├── configuration/           <- Store hydra variables and configuration data class.
    │
    ├── data_readers/            <- Objects to read the data and turn into View2D partitions
    │
    ├── preproccess/             <- Code to preprocess the dataset and extract the features for the pipeline
    │
    ├── models/                  <- The base models and any other methods needed to train the model.
    │
    ├── pipelines/               <- All the code responsible for the training pipeline 
    │
    ├── evaluation/              <- All the code responsible for the evaluation and inference 
    │
    ├── utils/                   <- All extra scripts
    │
    └── visualization/           <- Codes used to view and save graphical representations of data
```

## Configuration File

We will follow some rules for configuration files:

* The configuration file must be placed inside the root folder of the repository with the name `config.yaml`.

* By default, changes to the configuration file won't be accepted. If the structure of the config.yaml file is changed, then the new properties must be tracked

* The only way to send an updated config.yaml is in cases where new features and configurations become necessary. To do this, use:

```bash
git add --force config.yaml
```

* Don't send absolute or relative paths, connection strings or any other information that only exists on your machine. Assume that not everyone has the folder structure that you use.

* In the case of examples, you can look for examples used in the `src/configuration/examples` folder.
