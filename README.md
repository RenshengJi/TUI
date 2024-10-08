# TUI: A Conformal Uncertainty Indicator for Continual Test-Time Adaptation

Official code for  A Conformal Uncertainty Indicator for Continual Test-Time Adaptation.
Here, we use the CIFAR10-C dataset as an example to introduce the usage of the code.


## Prerequisite

Please create and activate the following conda envrionment. To reproduce our results, please kindly create and use this environment.

```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate tui
```


## Datasets

Please download the dataset [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk) to any path by yourself, and fill in the path in _C.DATA_DIR in conf.py.



## Test

```bash
python test_time.py --cfg ./cfgs/cifar10_c/CPCTTA.yaml --gpu 0 --CP_method "TUI" --CP_alpha 0.2
```
