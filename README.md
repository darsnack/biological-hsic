# Information Bottleneck-Based Hebbian Learning Rule Naturally Ties Working Memory and Synaptic Updates

A biologically plausible implementation of the HSIC objective.

## Setup instructions

To install the codebase, install Micromamba (or Conda), then run:
```shell
micromamba create -f environment.yaml
poetry install --no-root
```

## Running the experiments

The code for running the experiments is located in the root directory with
auxiliary code located in `projectlib`.
Configuration files use [Hydra](https://hydra.cc) and are found in `configs`.

### Small-scale experiments

To run the reservoir experiment, activate the project environment, and run:
```shell
python train-reservoir.py
```
The results will be stored under `outputs/train-reservoir`.

To run the linear synthetic dataset experiment, activate the project environment, and run:
```shell
python train-lif-biohsic.py data=linear
```
The results will be stored under `output/train-lif-biohsic`.

To run the XOR synthetic dataset experiment, activate the project environment, and run:
```shell
python train-lif-biohsic.py data=xor
```
The results will be stored under `output/train-lif-biohsic`.

### Large-scale experiments

To run the MNIST experiment, activate the project environment, and run:
```shell
python train-biohsic.py data=mnist
```
The results will be stored under `output/train-biohsic`.

To run the CIFAR-10 experiment, activate the project environment, and run:
```shell
python train-biohsic.py data=cifar10 model=cnn
```
The results will be stored under `output/train-biohsic`.

The back-propagation baselines can be run using the same commands but with the
`train-bp.py` script.

## Plot the results

The `plotting.ipynb` can recreate the plots given the results stored in the
output directories above. You may need to adjust the paths in the notebook to
your results.
