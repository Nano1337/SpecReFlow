# Getting Started

This document provides tutorials to train and evaluate SpecReFlow. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset preparation](DATA.md).

## Training

All training configurations for the Unet can be found in the [experiments](../experiments) folder. To train a model, first navigate to the lib directory with the train.py file. Then, run the following command (replace with correct absolute path to the config file):

```
python3 train.py --cfg /path/to/unet.yaml
```

## Reproducing Results

To ensure reliability of our results, we provide the following scripts to reproduce the figures and tables found in the paper. Please note that the results may vary slightly due to randomness in the training process and that paper figures were manually arranged in MS PowerPoint after images for those figures were generated using this code. 

### Reproducing Figures
To produce all figures, navigate to the 'src/lib' directory and run the following command:

```
python3 create_all_figs.py
```
The output images used in each figure will be found in the figs directory at the root of the project.

### Reproducing Tables
To produce all tables, navigate to the 'src/lib' directory and run the following command:

```
python3 create_all_tables.py
```
