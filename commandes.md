# Experiments commands

## Figure 1 - MNIST, LR, NOISE

```sh
nohup sh -c 'python main.py --lr 0.05 --field double-linear --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 0.1 --field double-linear --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 0.5 --field double-linear --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 0.05 --field double-exponential --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 0.1 --field double-exponential --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 0.5 --field double-exponential --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 5 --field double-linear --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 10 --field double-linear --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 50 --field double-linear --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 5 --field double-exponential --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 10 --field double-exponential --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 50 --field double-exponential --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 50 --field double-linear --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 100 --field double-linear --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 500 --field double-linear --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 50 --field double-exponential --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 100 --field double-exponential --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 500 --field double-exponential --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 750 --field double-linear --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 1500 --field double-linear --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 7500 --field double-linear --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 750 --field double-exponential --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 1500 --field double-exponential --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 7500 --field double-exponential --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 750 --field double-linear --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 1500 --field double-linear --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 7500 --field double-linear --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 750 --field double-exponential --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 1500 --field double-exponential --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 7500 --field double-exponential --n_models 10 --task MNIST --noise 0.015 && \
python main.py --lr 1250 --field double-linear --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 2500 --field double-linear --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 12500 --field double-linear --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 1250 --field double-exponential --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 2500 --field double-exponential --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 12500 --field double-exponential --n_models 10 --task MNIST --noise 0.05' &> nohup-mnist.txt &
```
