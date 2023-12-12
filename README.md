<a name="readme-top"></a>

# Project 2

This repository contains the work of Siim Markus Marvet and Jan Kokla 
for ML Project 2, where we aimed to separate roads from the background 
in satellite imagery.

## Getting Started

### Dependencies

Use the Pypi package manager to set up the environment.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### Extract Raw Data

Use the bash script below to download and extract the training and testing data.

```bash
sudo chmod +x scripts/extract-data.sh && ./scripts/extract-data.sh
```

### Download Final Models

TBD

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage

### Notebooks

The repository contains notebooks with experiments for finding the best possible model:

- [transformations.ipynb](notebooks/transformations.ipynb): compare different data augmentation techniques
- [decoders.ipynb](notebooks/decoders.ipynb): compare different decoders with fixed encoder
- [architectures.ipynb](notebooks/architectures.ipynb): benchmarks different encoder-decoder combinations
- [ensembling.ipynb](notebooks/ensembling.ipynb): compare majority voting with separate models
- [hyperopt.ipynb](notebooks/hyperopt.ipynb): tune hyperparameters with Flaml
- [final_training.ipynb](notebooks/final_training.ipynb): train the models on the whole dataset

### Inference

You can use `run.py` for running the inference. Notice that it is necessary to 
download the raw data and final models before running the script. 

Additionally, keep in mind that the experiments and inference was done on 
16GB NVIDIA V100 Tensor Core GPU and it might be unfeasible to run the script locally. 
For that purpose, we've set up a notebook that can be used in e.g. Google Colab.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

- Siim Markus Marvet - [siim.marvet@epfl.ch](mailto:siim.marvet@epfl.ch)
- Jan Kokla - [jan.kokla@epfl.ch](mailto:jan.kokla@epfl.ch)

<p align="right">(<a href="#readme-top">back to top</a>)</p>