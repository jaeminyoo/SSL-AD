# Data Augmentation is a Hyperparameter for Self-supervised Anomaly Detection

This is our implementation for [Data Augmentation is a Hyperparameter:
Cherry-picked Self-Supervision for Unsupervised Anomaly Detection is Creating
the Illusion of Success](https://arxiv.org/abs/2208.07734), published in
Transactions on Machine Learning Research (TMLR).

## Prerequisites

Our implementation is based on Python 3.8.12 and PyTorch 1.10.1. Refer to
`requirements.txt` for the required packages.

## How to Run

You can type the following command to train a denoising autoencoder (DAE) model
with a specified augmentation function, which is `flip` in this case:
```
cd src ; python main.py --data cifar10 --normal-class 0 --augment flip 
```
The `--normal-class` option chooses a class to consider as normal. The script
uses only the chosen class as training data, treating the rest as anomalous. The
training is done only once for each normal class, but the evaluation is done in
the one-vs-one scheme; we take the AUC value from each pair of classes and use
it to run the comprehensive Wilcoxon test.

The script automatically downloads the specified dataset and prints the
following log during the training (assuming that the dataset is already
downloaded):
```
Files already downloaded and verified
Files already downloaded and verified
Parameters: 4159235
[epoch   0] [0.0980] [0.4849]                                                                             
[epoch   1] [0.0748] [0.5054]                                                                             
[epoch   2] [0.0475] [0.6054]                                                                             
[epoch   3] [0.0358] [0.6607]                                                                             
[epoch   4] [0.0299] [0.6529]  
...
```
The first value at each line represents the training loss, which is the
reconstruction error, and the second value is the AUC measured in the
one-vs-rest scheme. No early stopping is used, since we have no validation data.
The result is stored at the `out-tmp` directory. You can change the arguments of
`main.py` to run experiments in other settings and configurations.
