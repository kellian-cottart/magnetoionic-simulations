import torch
from models import *
from optimizers import *
from dataloader import *
import tqdm
import json
import datetime
import argparse
from torch.nn.utils import parameters_to_vector

# Arguments
parser = argparse.ArgumentParser(
    description='Train a DNN with Magnetoionic optimizer')
parser.add_argument('--seed', type=int, default=100,
                    help='Seed for the random number generator')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for the training and testing')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs for the training')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate for the optimizer')
parser.add_argument('--field', type=str, default='strong', nargs='*',
                    help='Strength of the magnetic field (strong, weak or linear), can be strong weak to switch)')
parser.add_argument('--layers', type=int, nargs='+',
                    default=[512], help='Number of neurons in each hidden layer')
parser.add_argument('--scale', type=float, default=1,
                    help='Scale of the functions f_minus and f_plus')
parser.add_argument('--n_models', type=int, default=10,
                    help='Number of models to train')
parser.add_argument('--task', type=str, default='MNIST',
                    help='Task to perform (MNIST or Fashion)')
parser.add_argument('--time_switch', type=int, default=None,
                    help='Fraction of the epochs to switch the magnetic field (2 is half of the epochs)')
parser.add_argument('--init', type=float, default=0.01,
                    help='Standard deviation for the initialization of the weights')
parser.add_argument('--input_scale', type=float, default=0.001,
                    help='Scale of the input values')
parser.add_argument('--var', type=float, default=0.2,
                    help='Gaussian noise standard deviation to change the slope of the functions')
parser.add_argument('--clipping', type=float, default=0,
                    help='Clipping value for the gradients (lower bound)')
parser.add_argument('--resistor_noise', type=float, default=0,
                    help='Gaussian noise standard deviation to add to the resistor values')

args = parser.parse_args()

NUMBER_MODELS = args.n_models  # Number of models to train
PADDING = 2  # Padding for the MNIST dataset - N pixels on each side of the image
DEVICE = 'cuda:0'  # Device to use for the simulation (GPU or CPU)
SEED = args.seed  # Seed for the random number generator
INIT = args.init  # Standard deviation for the initialization of the weights
BATCH_SIZE = args.batch_size  # Batch size for the training and testing
EPOCHS = args.epochs  # Number of epochs for the training
LR = args.lr  # Learning rate for the optimizer
FIELD = args.field  # Strength of the magnetic field
DIVISOR = 1  # Divisor for the learning rate
SCALE = args.scale  # Scale of the functions f_minus and f_plus
INPUT_SCALE = args.input_scale  # Scale of the input values
# Gaussian noise standard deviation to add to the resistor values
RESISTOR_NOISE = args.resistor_noise
# Gaussian noise standard deviation to add to the voltage values
VOLTAGE_NOISE = RESISTOR_NOISE*1e-4
# Gaussian noise standard deviation to change the slope of the functions
DEVICE_VARIABILITY = args.var
CLIPPING = args.clipping  # Clipping value for the gradients (lower bound)
# Fraction of the epochs to switch the magnetic field
TIME_SWITCH = args.time_switch
TASK = args.task  # Task to perform (MNIST or Fashion)
LAYERS = args.layers  # Number of neurons in each hidden layer
LOSS = torch.nn.CrossEntropyLoss()  # Loss function
FOLDER = "simulations"  # Folder to save the simulations


def set_field(field):
    if field == "double-linear":
        def f(x): return -0.021*x + 2.742
        def f_inv(y): return (y - 2.742)/(-0.021)
    elif field == "double-exponential":
        def f(x): return 1.530 * torch.exp(-x/9.741) + 0.750
        def f_inv(y): return -9.741 * torch.log((y - 0.750)/1.530)
    return f, f_inv


def training(DEVICE, BATCH_SIZE, LOSS, LR, train_mnist, test_mnist, dnn, epochs, optim, pbar, switch=None, time_switch=2):
    accuracies = torch.zeros(epochs)
    gradients = torch.zeros(epochs)
    for epoch in pbar:
        if switch is not None and epoch == epochs // time_switch:
            optim.set_field(switch, SCALE)
        # TRAINING WITH BATCHES
        num_batches = len(train_mnist) // BATCH_SIZE + 1
        dnn.train()
        for batch in range(num_batches):
            start = batch * BATCH_SIZE
            end = start + BATCH_SIZE
            x, y = train_mnist[start:end]
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            yhat = dnn(x)
            l = LOSS(yhat, y)
            optim.zero_grad()
            l.backward()
            optim.step()
        # TEST WITH BATCHES
        accuracies[epoch] = evaluation(DEVICE, BATCH_SIZE, test_mnist, dnn)
        # Compute the mean absolute value of the gradients
        gradients[epoch] = LR*parameters_to_vector(
            [p.grad for p in dnn.parameters()]).abs().mean().item()
        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} - Test accuracy: {accuracies[epoch]*100:.2f} % - Loss: {l.item():.4f} - <|grad|>: {gradients[epoch]:.4f}")
    return l, accuracies, gradients


def evaluation(DEVICE, BATCH_SIZE, test_mnist, dnn):
    dnn.eval()
    num_batches = len(test_mnist) // BATCH_SIZE + 1
    acc = 0
    for batch in range(num_batches):
        start = batch * BATCH_SIZE
        end = start + BATCH_SIZE
        x, y = test_mnist[start:end]
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        yhat = dnn(x)
        acc += (yhat.argmax(dim=1) == y).sum().item()
    acc /= len(test_mnist)
    return acc


if __name__ == "__main__":
    simulation_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{TASK}-" + \
        "-".join(FIELD) + \
        f"-{LR}-var-{DEVICE_VARIABILITY}-clipping-{CLIPPING}-resistor-{RESISTOR_NOISE}-voltage-{VOLTAGE_NOISE}"
    os.makedirs(FOLDER, exist_ok=True)
    folder_path = os.path.join(FOLDER, simulation_id)
    for i in range(NUMBER_MODELS):
        print(f"Model {i+1}/{NUMBER_MODELS}")
        # SET THE SEED
        torch.manual_seed(SEED+i)
        torch.cuda.manual_seed_all(SEED+i)
        # LOAD MNIST
        loader = GPULoading(
            padding=PADDING,
            device=DEVICE,
        )
        train_mnist, test_mnist, shape, target_size = task_selection(
            loader=loader, batch_size=BATCH_SIZE, task=TASK)
        # MODEL DEFINITION
        size_in = torch.prod(torch.tensor(shape)).item()
        # OPTIMIZER
        field = FIELD[0]
        switch = None if len(FIELD) == 1 else FIELD[1]
        # Functions for the magnetic field
        f = set_field(field)[0]
        f_inv = set_field(field)[1]
        dnn = NNN(
            layers=[size_in] + LAYERS + [target_size],
            activation="relu",
            init="uniform",
            std=INIT,
            dropout=False,
            normalization="batchnorm",
            running_stats=True,
            device=DEVICE,
            input_scale=INPUT_SCALE,
            resistor_noise=RESISTOR_NOISE,
            voltage_noise=VOLTAGE_NOISE,
            f=f,
            f_inv=f_inv,
        )
        parameters = list(dnn.parameters())
        optim = MagnetoionicDouble(
            parameters,
            lr=LR,
            device_variability=DEVICE_VARIABILITY,
            clipping=CLIPPING,
            f=f,
            f_inv=f_inv,
        )
        pbar = tqdm.tqdm(range(EPOCHS))
        # TRAINING
        l, acc, grad = training(DEVICE, BATCH_SIZE, LOSS, LR, train_mnist,
                                test_mnist, dnn, EPOCHS, optim, pbar, switch=switch, time_switch=TIME_SWITCH)
        print(f"Accuracy: {acc[-1]*100: .2f} %, Loss: {l.item(): .4f}")

        ##############################
        os.makedirs(folder_path, exist_ok=True)
        # Save the model
        model_filename = f"{simulation_id}-{i}.pth"
        torch.save(dnn.state_dict(), os.path.join(folder_path, model_filename))
        # Save the accuracies
        acc_filename = f"{simulation_id}-{i}-accuracies.pth"
        torch.save(acc, os.path.join(folder_path, acc_filename))
        # Save the gradients
        grad_filename = f"{simulation_id}-{i}-gradients.pth"
        torch.save(grad, os.path.join(folder_path, grad_filename))
        params = list(dnn.parameters())
        for k in range(0, len(params), 2):
            x1 = f_inv(params[k].data)
            x2 = f_inv(params[k+1].data)
            f_exponential = set_field("double-exponential" if field ==
                                      "double-linear" else "double-linear")[0]
            params[k].data = f_exponential(x1)
            params[k+1].data = f_exponential(x2)

        bandwidth_linear = f(torch.tensor(0)).to(
            device=DEVICE) - f(torch.tensor(27)).to(device=DEVICE)
        bandwidth_exp = f_exponential(
            torch.tensor(0)).to(device=DEVICE) - f_exponential(torch.tensor(27)).to(device=DEVICE)
        ratio = bandwidth_linear / \
            bandwidth_exp if field == "double-linear" else bandwidth_exp/bandwidth_linear
        dnn.set_input_scale(dnn.input_scale*ratio)

        swap_eval_acc = evaluation(DEVICE, BATCH_SIZE, test_mnist, dnn)
        # Save the simulation parameters in a dict
        simulation_parameters = {
            "task": TASK,
            "seed": SEED,
            "padding": PADDING,
            "batch_size": BATCH_SIZE,
            "layers": LAYERS,
            "epochs": EPOCHS,
            "input_scale": INPUT_SCALE,
            "voltage_noise": VOLTAGE_NOISE,
            "resistor_noise": RESISTOR_NOISE,
            "time_switch": TIME_SWITCH,
            "init": INIT,
            "optimizer": optim.__class__.__name__,
            "optimizer_parameters": {
                "lr": LR,
                "field": FIELD,
                "scale": SCALE,
                "device_variability": DEVICE_VARIABILITY,
                "clipping": CLIPPING,
            },
            "criterion": LOSS.__class__.__name__,
            "loss": l.item(),
            "accuracy": acc[-1].item()*100,
            "swap_accuracy": swap_eval_acc*100,
        }
        # Save the simulation parameters in a json file
        filename = f"{simulation_id}-{i}.json"
        with open(os.path.join(folder_path, filename), 'w') as f:
            json.dump(simulation_parameters, f, indent=4)
    print(f"Simulation saved in {folder_path}")
