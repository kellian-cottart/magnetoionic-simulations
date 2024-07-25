# Experiments commands

## Figure 1 - MNIST, LR, NOISE

We model the noise of the reading procedure by a Gaussian noise $X \sim \mathcal{N}(0, \sigma_{\text{noise}})$. To match experimental conditions, we take $\sigma_{\text{noise}} = 0.015$. This measure noise is dictated by how much variation there is when trying to get the value of a weight given a certain input.

We model the intrinsic device variability by a Gaussian noise $Y \sim \mathcal{N}(1, \sigma_{\text{device}})$. To match experimental conditions, we take $\sigma_{\text{device}} = 0.2$. This device variability corresponds to the different slopes of the functions of each device.

We will run experiments on MNIST dataset using `double-linear` and `double-exponential` devices and see which one performs best. The learning rate we will use are 100, 250, 500, 1000, 2500, 5000, and 10000.

```sh
nohup sh -c 'python main.py --lr 100 --field double-linear --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 100 --field double-exponential --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 250 --field double-linear --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 250 --field double-exponential --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 500 --field double-linear --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 500 --field double-exponential --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 1000 --field double-linear --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 1000 --field double-exponential --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 2500 --field double-linear --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 2500 --field double-exponential --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 5000 --field double-linear --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 5000 --field double-exponential --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 10000 --field double-linear --task MNIST --noise 0.015 --var 0.2 && \
python main.py --lr 10000 --field double-exponential --task MNIST --noise 0.015 --var 0.2' &> figure-1.txt &
```

## Figure 2 - MNIST, LR, NOISE, CLIPPING

```sh
nohup sh -c 'python main.py --lr 100 --field double-linear --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 100 --field double-exponential --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 250 --field double-linear --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 250 --field double-exponential --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 500 --field double-linear --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 500 --field double-exponential --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 1000 --field double-linear --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 1000 --field double-exponential --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 2500 --field double-linear --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 2500 --field double-exponential --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 5000 --field double-linear --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 5000 --field double-exponential --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 10000 --field double-linear --task MNIST --noise 0.015 --var 0.2 --clipping 0.1 && \
python main.py --lr 10000 --field double-exponential --task MNIST --noise 0.015 --var 0.2 --clipping 0.1' &> figure-2.txt &
```
