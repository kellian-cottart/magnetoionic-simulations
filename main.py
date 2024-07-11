import torch
from models import DNN
from optimizers import Magnetoionic
from dataloader import *
import tqdm
import json
import datetime
import argparse
import argparse

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
parser.add_argument('--n_models', type=int, default=5,
                    help='Number of models to train')
parser.add_argument('--task', type=str, default='MNIST',
                    help='Task to perform (MNIST or Fashion)')
parser.add_argument('--time_switch', type=int, default=2,
                    help='Fraction of the epochs to switch the magnetic field (2 is half of the epochs)')

args = parser.parse_args()

NUMBER_MODELS = args.n_models  # Number of models to train
PADDING = 2  # Padding for the MNIST dataset - N pixels on each side of the image
DEVICE = 'cuda:0'  # Device to use for the simulation (GPU or CPU)
SEED = args.seed  # Seed for the random number generator
BATCH_SIZE = args.batch_size  # Batch size for the training and testing
EPOCHS = args.epochs  # Number of epochs for the training
LR = args.lr  # Learning rate for the optimizer
FIELD = args.field  # Strength of the magnetic field
DIVISOR = 1  # Divisor for the learning rate
SCALE = args.scale  # Scale of the functions f_minus and f_plus
# Fraction of the epochs to switch the magnetic field
TIME_SWITCH = args.time_switch
TASK = args.task  # Task to perform (MNIST or Fashion)
LAYERS = args.layers  # Number of neurons in each hidden layer
LOSS = torch.nn.CrossEntropyLoss()  # Loss function
FOLDER = "simulations"  # Folder to save the simulations


def training(DEVICE, BATCH_SIZE, LOSS, train_mnist, test_mnist, dnn, epochs, optim, pbar, switch=None, time_switch=2):
    accuracies = torch.zeros(epochs)
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
        acc = evaluation(DEVICE, BATCH_SIZE, test_mnist,
                         dnn, accuracies, epoch)
        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} - Test accuracy: {acc*100:.2f} % - Loss: {l.item():.4f}")
    return l, accuracies


def evaluation(DEVICE, BATCH_SIZE, test_mnist, dnn, accuracies, epoch):
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
    accuracies[epoch] = acc
    return acc


if __name__ == "__main__":
    if isinstance(FIELD, list):
        simulation_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{TASK}-" + \
            "-".join(FIELD) + f"-{LR}-switch-{TIME_SWITCH}"
    else:
        simulation_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{TASK}-{FIELD}-{LR}"
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
        dnn = DNN(
            layers=[size_in] + LAYERS + [target_size],
            activation="relu",
            init="uniform",
            std=0.01,
            dropout=False,
            normalization="batchnorm",
            running_stats=True,
            device=DEVICE
        )
        # OPTIMIZER
        field = FIELD if isinstance(FIELD, str) else FIELD[0]
        switch = None if isinstance(FIELD, str) else FIELD[1]
        optim = Magnetoionic(dnn.parameters(), lr=LR, field=field, scale=SCALE)
        pbar = tqdm.tqdm(range(EPOCHS))
        # TRAINING
        l, acc = training(DEVICE, BATCH_SIZE, LOSS, train_mnist,
                          test_mnist, dnn, EPOCHS, optim, pbar, switch=switch, time_switch=TIME_SWITCH)
        print(f"Accuracy: {acc[-1]*100: .2f} %, Loss: {l.item(): .4f}")
        # SAVE THE SIMULATION
        # Save the simulation parameters in a dict
        simulation_parameters = {
            "task": TASK,
            "seed": SEED,
            "padding": PADDING,
            "batch_size": BATCH_SIZE,
            "layers": LAYERS,
            "epochs": EPOCHS,
            "optimizer": optim.__class__.__name__,
            "optimizer_parameters": {
                "lr": LR,
                "field": FIELD
            },
            "criterion": LOSS.__class__.__name__,
            "loss": l.item(),
            "accuracy": acc[-1].item()*100,
        }
        # Save the simulation parameters in a json file
        filename = f"{simulation_id}-{i}.json"
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, filename), 'w') as f:
            json.dump(simulation_parameters, f, indent=4)
        # Save the model
        model_filename = f"{simulation_id}-{i}.pth"
        torch.save(dnn.state_dict(), os.path.join(folder_path, model_filename))
        # Save the accuracies
        acc_filename = f"{simulation_id}-{i}-accuracies.pth"
        torch.save(acc, os.path.join(folder_path, acc_filename))
    print(f"Simulation saved in {folder_path}")
