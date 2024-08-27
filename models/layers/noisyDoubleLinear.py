import torch


class NoisyDoubleLinear(torch.nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            f,
            f_inv,
            resistor_noise=0.0,
            voltage_noise=0.0,
            input_scale=0.001,
            init=0.01,
            device_noise=False,
            device="cuda:0"):
        super(NoisyDoubleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.resistor_noise = resistor_noise
        self.voltage_noise = voltage_noise
        self.device_noise = device_noise
        self.init = init
        self.device = device
        self.input_scale = input_scale
        self.f = f
        self.f_inv = f_inv
        self.weight1 = torch.nn.Parameter(torch.empty(
            (out_features, in_features), device=self.device))
        self.weight2 = torch.nn.Parameter(torch.empty(
            (out_features, in_features), device=self.device))
        self.reset_parameters()

    def reset_parameters(self):
        self.start = self.f(torch.tensor(0)).to(self.device)
        self.stop = self.f(torch.tensor(27)).to(self.device)
        self.weight1.data = self.start * \
            torch.ones_like(self.weight1).to(self.device) - self.init
        self.weight2.data = self.start * torch.ones_like(self.weight2).to(
            self.device) - self.init - torch.empty_like(self.weight2).uniform_(-self.init, self.init).to(self.device)

    def forward(self, x):
        intensity = x * self.input_scale
        voltage = NoisyForward.apply(
            intensity, self.weight1, self.weight2, self.resistor_noise, self.device_noise)
        voltage_noise = torch.empty_like(voltage).normal_(
            0, self.voltage_noise).to(self.device)
        return voltage + voltage_noise

    def __repr__(self):
        return f"NoisyDoubleLinear(in_features={self.in_features}, out_features={self.out_features}, resistor_noise={self.resistor_noise}, voltage_noise={self.voltage_noise}, input_scale={self.input_scale}, init={self.init}, device={self.device})"


class NoisyForward(torch.autograd.Function):
    def forward(ctx, x, weight1, weight2, noise, device_noise=False):
        """Forward pass of the noisy double linear layer
        """
        ctx.save_for_backward(x, weight1, weight2)
        ctx.noise = noise
        if device_noise == True:
            noise1 = torch.empty_like(weight1).normal_(
                0, ctx.noise).to(weight1.device)
            noise2 = torch.empty_like(weight2).normal_(
                0, ctx.noise).to(weight2.device)
            resistor = weight2 - weight1 + noise2 - noise1
        else:
            resistor = weight2 - weight1
        intensity = x
        return torch.functional.F.linear(intensity, resistor)

    def backward(ctx, grad_output):
        """Backward pass of the noisy double linear layer
        """
        x, weight1, weight2 = ctx.saved_tensors
        noise1 = torch.empty_like(weight1).normal_(
            0, ctx.noise).to(weight1.device)
        noise2 = torch.empty_like(weight2).normal_(
            0, ctx.noise).to(weight2.device)
        resistor = weight2 - weight1 + noise2 - noise1
        grad_input = grad_output @ resistor
        grad_weight1 = -grad_output.T @ x
        grad_weight2 = grad_output.T @ x
        return grad_input, grad_weight1, grad_weight2, None, None
