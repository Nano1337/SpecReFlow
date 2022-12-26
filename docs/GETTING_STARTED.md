# Getting Started

This document provides tutorials to train and evaluate SpecReFlow. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset preparation](DATA.md).

## Training

All training configurations for the Unet can be found in the [experiments](../experiments) folder. To train a model, first navigate to the lib directory with the train.py file. Then, run the following command (replace with correct absolute path to the config file):

```
python3 train.py --cfg /path/to/unet.yaml
```