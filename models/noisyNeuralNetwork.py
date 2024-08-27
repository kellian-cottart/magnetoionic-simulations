import torch
from .layers import *


class NNN(torch.nn.Module):
    """ Deep Neural Network

    Args:
        f (callable): Forward function for the weights
        f_inv (callable): Inverse function for the weights
        layers (list): List of layer sizes (including input and output layers)
        init (str): Initialization method for weights
        std (float): Standard deviation for initialization
        device (str): Device to run on
        dropout (bool): Whether to use dropout
        normalization (str): Normalization layer type
        bias (bool): Whether to use bias
        running_stats (bool): Whether to use running stats in normalization layer
        affine (bool): Whether to use affine transformation in normalization layer
        eps (float): Epsilon for normalization layer
        momentum (float): Momentum for normalization layer
        activation_function (str): Activation function
        input_scale (float): Input scale
        resistor_noise (float): Resistor noise
        voltage_noise (float): Voltage noise
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
    """

    def __init__(self,
                 f: callable,
                 f_inv: callable,
                 layers: list = [1024, 1024],
                 init: str = "uniform",
                 std: float = 0.01,
                 device: str = "cuda:0",
                 dropout: bool = False,
                 normalization: str = None,
                 bias: bool = False,
                 running_stats: bool = False,
                 affine: bool = False,
                 eps: float = 1e-5,
                 momentum: float = 0.15,
                 activation_function: str = "relu",
                 input_scale: float = 0.001,
                 resistor_noise=0.0,
                 voltage_noise=0.0,
                 device_noise=False,
                 * args,
                 **kwargs):
        super(NNN, self).__init__()
        self.device = device
        self.layers = torch.nn.ModuleList().to(self.device)
        self.dropout = dropout
        self.normalization = normalization
        self.eps = eps
        self.momentum = momentum
        self.running_stats = running_stats
        self.affine = affine
        self.activation_function = activation_function
        self.input_scale = input_scale
        self.resistor_noise = resistor_noise
        self.voltage_noise = voltage_noise
        self.device_noise = device_noise
        self.f = f
        self.f_inv = f_inv

        if "activation_parameters" in kwargs:
            self.activation_parameters = kwargs["activation_parameters"]
        ### LAYER INITIALIZATION ###
        self._layer_init(layers, bias)
        ### WEIGHT INITIALIZATION ###
        self._weight_init(init, std)

    def set_input_scale(self, input_scale):
        """ Set input scale for the forward pass

        Args:
            input_scale (float): Input scale
        """
        self.input_scale = input_scale
        for layer in self.layers:
            if isinstance(layer, NoisyDoubleLinear):
                layer.input_scale = input_scale

    def _layer_init(self, layers, bias=False):
        """ Initialize layers of NN

        Args:
            layers (list): List of layer sizes (including input and output layers)
            bias (bool): Whether to use bias
        """
        self.layers.append(torch.nn.Flatten().to(self.device))
        for i, _ in enumerate(layers[:-1]):
            # Linear layers with BatchNorm
            if self.dropout and i != 0:
                self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(NoisyDoubleLinear(
                layers[i],
                layers[i+1],
                f=self.f,
                f_inv=self.f_inv,
                resistor_noise=self.resistor_noise,
                voltage_noise=self.voltage_noise,
                input_scale=self.input_scale,
                device_noise=self.device_noise,
                device=self.device))
            self.layers.append(self._norm_init(layers[i+1]))
            if i < len(layers)-2:
                self.layers.append(self._activation_init())

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        ### FORWARD PASS ###
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_with_intermediate(self, x, *args, **kwargs):
        """ Forward pass of DNN with intermediate outputs for the voltage and intensity

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
            list: List of input intensities
            list: List of output voltages
        """
        ### FORWARD PASS ###
        intensity = []
        voltage = []
        for layer in self.layers:
            if isinstance(layer, NoisyDoubleLinear):
                intensity.append(x*self.input_scale)
            x = layer(x)
            if isinstance(layer, NoisyDoubleLinear):
                voltage.append(x)
        return x, intensity, voltage

    def _activation_init(self):
        """
        Returns:
            torch.nn.Module: Activation function module
        """
        activation_functions = {
            "relu": torch.nn.ReLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "tanh": torch.nn.Tanh,
        }
        # add parameters to activation function if needed
        try:
            return activation_functions.get(self.activation_function, torch.nn.Identity)(**self.activation_parameters).to(self.device)
        except:
            return activation_functions.get(self.activation_function, torch.nn.Identity)().to(self.device)

    def _norm_init(self, n_features):
        """
        Args:
            n_features (int): Number of features

        Returns:
            torch.nn.Module: Normalization layer module
        """
        normalization_layers = {
            "batchnorm": lambda: torch.nn.BatchNorm1d(n_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.running_stats),
            "instancenorm": lambda: torch.nn.InstanceNorm1d(n_features, eps=self.eps, affine=self.affine, track_running_stats=self.running_stats),
        }
        return normalization_layers.get(self.normalization, torch.nn.Identity)().to(self.device)

    def _weight_init(self, init='normal', std=0.1):
        """ Initialize weights of each layer

        Args:
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
        """
        for layer in self.layers:
            if isinstance(layer, torch.nn.Module) and hasattr(layer, 'weight') and layer.weight is not None:
                if init == 'gaussian':
                    torch.nn.init.normal_(
                        layer.weight.data, mean=0.0, std=std)
                elif init == 'uniform':
                    torch.nn.init.uniform_(
                        layer.weight.data, a=-std/2, b=std/2)
                elif init == 'xavier':
                    torch.nn.init.xavier_normal_(layer.weight.data)
