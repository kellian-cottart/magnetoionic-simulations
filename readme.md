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

Simulations are saved under the `simulations` folder, with the timestamp corresponding with the start of the simulation. There are three types of files:

- `-accuracies.pth` that contains the accuracies obtained at each epoch in the torch format;
- `-gradients.pth` that contains the mean absolute gradients obtained at each epoch in the torch format;
- `.json` that contains the settings for the simulation;
- `.pth` that contains the weights of the model.

The argument `--n_models 10` is used to retrieve these files for the 10 models trained for each simulation.

Figures are computed in the notebook `magnetoionic-figures.ipynb`. All figures are exported in the folder `figures` under their corresponding timestamp.

## Description

### Objective

The main objective of the simulation is to show that learning with two devices is better when these two devices are linear rather than exponential.

We consider a double device per weight system where $w = w_2 - w_1$. $w_1, w_2$ are the values obtained for each device by applying a given pulse, and retrieving the Hall voltage. We have two different systems, `linear` and `exponential` corresponding to the following equations:

$$
\begin{align*}
    \text{Linear:} & \quad w_i = -0.021x + 2.742 \\
    \text{Exponential:} & \quad w_i = 1.530\exp\left(-\frac{x}{9.741}\right) + 0.750
\end{align*}
$$

where $x$ is the pulse applied to the device. The effective range is $x \in [0, 27]$ due to measure limitations.

### Description of the noise

We consider that the inference is done on hardware for the layers that have learnable weights. We can retrieve a Hall voltage $V = R^TI$ that will be our matrix multiplication plus some Gaussian reading noise with standard deviation $\sigma_{\text{r}}$ for each reading. When performing the backward pass, we read the values of the Magnetoionic devices to compute the gradient on computer, which also has a Gaussian reading noise $\sigma_{\text{v}}$ for each reading. Furthermore, the devices also have intrisic variability, that we model by a Gaussian noise $\sigma_{\text{var}}$, leading to perturbations in the functions used to backpropagate.

We model the noise of the reading procedure on the devices by a Gaussian noise $\epsilon_r\sim\mathcal{N}(0, \sigma_{\text{r}})$ for the weights, and by $\epsilon_v\sim\mathcal{N}(0, \sigma_{\text{v}})$ for the matrix multiplication/voltage. To match experimental conditions, we take $\sigma_{\text{r}} = 0.015$ and $\sigma_{\text{v}} = 1.5\times10^{-6}$.

We model the intrinsic device variability by a Gaussian noise $\epsilon_\text{var}\sim\mathcal{N}(1, \sigma_{\text{device}})$ that will multiply the gradient during the backward pass. This means that the corresponding weight for a given pulse will be always different for each device. To match experimental conditions, we take $\sigma_{\text{device}} = 0.2$. This device variability corresponds to the different slopes of the functions of each device.

We will run experiments on MNIST dataset using `double-linear` and `double-exponential` devices and see which one performs best. The learning rate we will use are 500, 1000, 2000, 3000, 4000, 5000.

## Figures

### Figure 1 - Clipping 0.1, Noise 0.015, Var 0.2

```sh
nohup sh -c 'python main.py --lr 500 --field double-linear --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1  && \
python main.py --lr 1000 --field double-linear --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1  && \
python main.py --lr 2000 --field double-linear --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1  && \
python main.py --lr 3000 --field double-linear --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1  && \
python main.py --lr 4000 --field double-linear --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1  && \
python main.py --lr 5000 --field double-linear --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1 && \
python main.py --lr 500 --field double-exponential --task MNIST --var 0.2 --resistor_noise 0.015 ---clipping 0.1 && \
python main.py --lr 1000 --field double-exponential --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1 && \
python main.py --lr 2000 --field double-exponential --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1 && \
python main.py --lr 3000 --field double-exponential --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1 && \
python main.py --lr 4000 --field double-exponential --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1 && \
python main.py --lr 5000 --field double-exponential --task MNIST --var 0.2 --resistor_noise 0.015 --clipping 0.1'&> figure1.txt &
```

### Figure 2 - Clipping 0, Noise 0, Var 0

```sh
nohup sh -c'python main.py --lr 500 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0 && \
python main.py --lr 1000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0 && \
python main.py --lr 2000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0  && \
python main.py --lr 3000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0  && \
python main.py --lr 4000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0  && \
python main.py --lr 5000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0 && \
python main.py --lr 500 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0 && \
python main.py --lr 1000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0 && \
python main.py --lr 2000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0 && \
python main.py --lr 3000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0 && \
python main.py --lr 4000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0 && \
python main.py --lr 5000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0'&> figure2.txt &
```

### Figure 3 - Clipping 0.1, 0.2, 0.5

```sh
nohup sh -c'python main.py --lr 500 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.1 && \
python main.py --lr 1000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.1 && \
python main.py --lr 2000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.1  && \
python main.py --lr 3000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.1  && \
python main.py --lr 4000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.1  && \
python main.py --lr 5000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.1 && \
python main.py --lr 500 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.1 && \
python main.py --lr 1000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.1 && \
python main.py --lr 2000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.1 && \
python main.py --lr 3000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.1 && \
python main.py --lr 4000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.1 && \
python main.py --lr 5000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.1 && \
python main.py --lr 500 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.2 && \
python main.py --lr 1000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.2 && \
python main.py --lr 2000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.2  && \
python main.py --lr 3000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.2  && \
python main.py --lr 4000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.2  && \
python main.py --lr 5000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.2 && \
python main.py --lr 500 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.2 && \
python main.py --lr 1000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.2 && \
python main.py --lr 2000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.2 && \
python main.py --lr 3000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.2 && \
python main.py --lr 4000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.2 && \
python main.py --lr 5000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.2 && \
python main.py --lr 500 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.5 && \
python main.py --lr 1000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.5 && \
python main.py --lr 2000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.5  && \
python main.py --lr 3000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.5  && \
python main.py --lr 4000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.5  && \
python main.py --lr 5000 --field double-linear --task MNIST --var 0 --resistor_noise 0 --clipping 0.5 && \
python main.py --lr 500 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.5 && \
python main.py --lr 1000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.5 && \
python main.py --lr 2000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.5 && \
python main.py --lr 3000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.5 && \
python main.py --lr 4000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.5 && \
python main.py --lr 5000 --field double-exponential --task MNIST --var 0 --resistor_noise 0 --clipping 0.5'&> figure3.txt &
```

## Authors

- Guillaume BERNARD (C2N/CNRS)
- Kellian COTTART (C2N/CNRS)
- Liza Herrera DIEZ (C2N/CNRS)
