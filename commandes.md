# Experiments commands

## Figure 1 - MNIST, LR, NOISE

Lr = {0, 0.5, 1, 5, 10, 50}, noise = {0, 0.001, 0.005, 0.01, 0.05}

```sh
nohup sh -c 'python main.py --lr 0.5 --noise 0 --task MNIST --field double-linear && \
python main.py --lr 0.5 --noise 0.001 --task MNIST --field double-linear && \
python main.py --lr 0.5 --noise 0.005 --task MNIST --field double-linear && \
python main.py --lr 0.5 --noise 0.01 --task MNIST --field double-linear && \
python main.py --lr 0.5 --noise 0.05 --task MNIST --field double-linear && \
python main.py --lr 1 --noise 0 --task MNIST --field double-linear && \
python main.py --lr 1 --noise 0.001 --task MNIST --field double-linear && \
python main.py --lr 1 --noise 0.005 --task MNIST --field double-linear && \
python main.py --lr 1 --noise 0.01 --task MNIST --field double-linear && \
python main.py --lr 1 --noise 0.05 --task MNIST --field double-linear && \
python main.py --lr 5 --noise 0 --task MNIST --field double-linear && \
python main.py --lr 5 --noise 0.001 --task MNIST --field double-linear && \
python main.py --lr 5 --noise 0.005 --task MNIST --field double-linear && \
python main.py --lr 5 --noise 0.01 --task MNIST --field double-linear && \
python main.py --lr 5 --noise 0.05 --task MNIST --field double-linear && \
python main.py --lr 10 --noise 0 --task MNIST --field double-linear && \
python main.py --lr 10 --noise 0.001 --task MNIST --field double-linear && \
python main.py --lr 10 --noise 0.005 --task MNIST --field double-linear && \
python main.py --lr 10 --noise 0.01 --task MNIST --field double-linear && \
python main.py --lr 10 --noise 0.05 --task MNIST --field double-linear && \
python main.py --lr 50 --noise 0 --task MNIST --field double-linear && \
python main.py --lr 50 --noise 0.001 --task MNIST --field double-linear && \
python main.py --lr 50 --noise 0.005 --task MNIST --field double-linear && \
python main.py --lr 50 --noise 0.01 --task MNIST --field double-linear && \
python main.py --lr 50 --noise 0.05 --task MNIST --field double-linear && \ 
python main.py --lr 0.5 --noise 0 --task MNIST --field double-exponential && \
python main.py --lr 0.5 --noise 0.001 --task MNIST --field double-exponential && \
python main.py --lr 0.5 --noise 0.005 --task MNIST --field double-exponential && \
python main.py --lr 0.5 --noise 0.01 --task MNIST --field double-exponential && \
python main.py --lr 0.5 --noise 0.05 --task MNIST --field double-exponential && \
python main.py --lr 1 --noise 0 --task MNIST --field double-exponential && \
python main.py --lr 1 --noise 0.001 --task MNIST --field double-exponential && \
python main.py --lr 1 --noise 0.005 --task MNIST --field double-exponential && \
python main.py --lr 1 --noise 0.01 --task MNIST --field double-exponential && \
python main.py --lr 1 --noise 0.05 --task MNIST --field double-exponential && \
python main.py --lr 5 --noise 0 --task MNIST --field double-exponential && \
python main.py --lr 5 --noise 0.001 --task MNIST --field double-exponential && \
python main.py --lr 5 --noise 0.005 --task MNIST --field double-exponential && \
python main.py --lr 5 --noise 0.01 --task MNIST --field double-exponential && \
python main.py --lr 5 --noise 0.05 --task MNIST --field double-exponential && \
python main.py --lr 10 --noise 0 --task MNIST --field double-exponential && \
python main.py --lr 10 --noise 0.001 --task MNIST --field double-exponential && \
python main.py --lr 10 --noise 0.005 --task MNIST --field double-exponential && \
python main.py --lr 10 --noise 0.01 --task MNIST --field double-exponential && \
python main.py --lr 10 --noise 0.05 --task MNIST --field double-exponential && \
python main.py --lr 50 --noise 0 --task MNIST --field double-exponential && \
python main.py --lr 50 --noise 0.001 --task MNIST --field double-exponential && \
python main.py --lr 50 --noise 0.005 --task MNIST --field double-exponential && \
python main.py --lr 50 --noise 0.01 --task MNIST --field double-exponential && \
python main.py --lr 50 --noise 0.05 --task MNIST --field double-exponential' &> mnist_lr_noise.txt &
```

## Figure 2 - MNIST, LR, VARIABILITY
```sh
nohup sh -c 'python main.py --lr 0.5 --noise 0 --var 0.2 --task MNIST --field double-linear && \
python main.py --lr 1 --noise 0 --var 0.2 --task MNIST --field double-linear && \
python main.py --lr 5 --noise 0 --var 0.2 --task MNIST --field double-linear && \
python main.py --lr 10 --noise 0 --var 0.2 --task MNIST --field double-linear && \
python main.py --lr 50 --noise 0 --var 0.2 --task MNIST --field double-linear && \
python main.py --lr 0.5 --noise 0 --var 0.2 --task MNIST --field double-exponential && \
python main.py --lr 1 --noise 0 --var 0.2 --task MNIST --field double-exponential && \
python main.py --lr 5 --noise 0 --var 0.2 --task MNIST --field double-exponential && \
python main.py --lr 10 --noise 0 --var 0.2 --task MNIST --field double-exponential && \
python main.py --lr 50 --noise 0 --var 0.2 --task MNIST --field double-exponential' &> mnist_var.txt &
```
