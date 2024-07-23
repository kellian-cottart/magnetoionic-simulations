# Let's do noise = {0, 0.001, 0.005, 0.01, 0.05} and lr = {0.5, 1, 5, 10, 50, 100} for both double-exponential and double-linear
```sh
nohup sh -c 'python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0 && \
python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.001 && \
python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.005 && \
python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.01 && \
python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.05 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task MNIST --noise 0.05' &> nohup-mnist.txt &
```

# Let's do noise = {0, 0.001, 0.005, 0.01, 0.05} and lr = {0.5, 1, 5, 10, 50, 100} for both double-exponential and double-linear
```sh
nohup sh -c 'python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0 && \
python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.001 && \
python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.005 && \
python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.01 && \
python main.py --lr 0.5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 1 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 5 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 10 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 50 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 100 --field double-exponential --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 0.5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 1 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 5 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 10 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 50 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.05 && \
python main.py --lr 100 --field double-linear --layers 512 --n_models 10 --task Fashion --noise 0.05' &> nohup-Fashion.txt &
```


```sh
nohup sh -c 'python main.py --lr 1 --field double-linear --n_models 10 --task MNIST --noise 1e-5 --input_scale 0.001 --output_scale 100 && \
python main.py --lr 1 --field double-linear --n_models 10 --task MNIST --noise 1e-4 --input_scale 0.001 --output_scale 100 && \
python main.py --lr 1 --field double-linear --n_models 10 --task MNIST --noise 1e-3 --input_scale 0.001 --output_scale 100 && \
python main.py --lr 1 --field double-exponential --n_models 10 --task MNIST --noise 1e-5 --input_scale 0.001 --output_scale 100 && \
python main.py --lr 1 --field double-exponential --n_models 10 --task MNIST --noise 1e-4 --input_scale 0.001 --output_scale 100 && \
python main.py --lr 1 --field double-exponential --n_models 10 --task MNIST --noise 1e-3 --input_scale 0.001 --output_scale 100' &> nohup-test.txt &
```
