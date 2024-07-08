# Repository for simulation data

## Requirements

To run the project, the required packages are:

- Python >= 3.9.18
- Pytorch >= 2.0
- Torchvision
- Matplotlib
- Cuda >= 12.0
- idx2numpy
- tqdm

A conda environment is provided in the `environment.yml` file. To install the environment `machine-learning-env`, run:

```bash
conda env create -f environment.yml 
```

## Description

To run the simulations, the file `main.py` contains all necessary elements. The different field simulations can be obtained by running the following commands.

For the strong magnetic field:

```bash
python main.py --lr 0.01 --field strong --layers 512 --n_models 5 --scale 1
```

For the weak magnetic field:

```bash
python main.py --lr 0.01 --field weak --layers 512 --n_models 5  --scale 1
```

For the linear simulation (without field):

```bash
python main.py --lr 0.01 --field linear --layers 512 --n_models 5 --scale 1
```

Note: Simulations are saved under the `simulations` folder, with the timestamp corresponding with the start of the simulation. There are three types of files:

- `-accuracies.pth` that contains the accuracies obtained at each epoch (to plot the results) in the torch format;
- `.json` that contains the settings for the simulation;
- `.pth` that contains the weights of the model.

When running simulations with multiple networks, one can expect n times these files.

## Figures

Figures are computed in the notebook `magnetoionic-figures.ipynb`, for now there are only two figures:

- The functions used to represent the magnetic fields;
- The accuracies, plotted and exported as both `.pdf` and `.svg`.

All figures are exported in the folder `figures` under their corresponding timestamp.

## Authors

- Liza Herrera DIEZ (C2N/CNRS)
- Guillaume BERNARD (C2N/CNRS)
- Kellian COTTART (C2N/CNRS)
