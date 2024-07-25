import torch


class MagnetoionicDouble(torch.optim.Optimizer):
    """ Magnetoionic Optimizer made for simulating magnetic fields in magnetoinic devices

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        field (str, optional): type of magnetic field applied, either "strong", "weak" or "linear"
        scale (float, optional): scale of the functions f_minus and f_plus
        noise (float, optional): Gaussian noise standard deviation to add to the gradients
        device_variability (float, optional): Gaussian noise standard deviation to change the slope of the functions
        eps (float, optional): term added to the denominator to improve numerical stability
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 field="double-linear",
                 device_variability=0.2,
                 clipping=0.1,
                 eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        self.set_field(field)
        defaults = dict(lr=lr,
                        eps=eps,
                        field=field,
                        device_variability=device_variability,
                        clipping=clipping)
        super(MagnetoionicDouble, self).__init__(params, defaults)

    def set_field(self, field):
        if field == "double-linear":
            self.f = lambda x: -0.021*x + 2.742
            self.f_inv = lambda y: (y - 2.742)/(-0.021)
        elif field == "double-exponential":
            self.f = lambda x: 1.530 * torch.exp(-x/9.741) + 0.750
            self.f_inv = lambda y: -9.741 * torch.log((y - 0.750)/1.530)
        self.field = field

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure(callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for i in range(0, len(group['params']), 2):
                w1 = group['params'][i]
                w2 = group['params'][i+1]
                start = self.f(torch.tensor(0)).to(
                    w1.data.device)
                stop = self.f(torch.tensor(27)).to(
                    w1.data.device)
                state = self.state[w1]
                lr = group['lr']
                variability = group['device_variability']
                clipping = group["clipping"]
                grad_w1 = w1.grad
                grad_w2 = w2.grad
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                state['step'] += 1
                if state['step'] == 1:
                    # Noise to multiply the gradient by to introduce variability in the device
                    state[f'variability_w1_{i}'] = torch.empty_like(
                        w1).normal_(1, variability).abs()
                    state[f'variability_w2_{i}'] = torch.empty_like(
                        w2).normal_(1, variability).abs()
                w1.data = torch.clamp(w1.data, stop, start)
                w2.data = torch.clamp(w2.data, stop, start)
                # Retrieve the number of pulses associated with the current weight state
                x1 = self.f_inv(w1.data)
                x2 = self.f_inv(w2.data)
                # Compute the new value of the weights given a pulse set as the gradient
                update1 = lr*torch.abs(grad_w1)*state[f'variability_w1_{i}']
                update2 = lr*torch.abs(grad_w2)*state[f'variability_w2_{i}']
                # Clip updates
                update1 = torch.where(update1 < clipping,
                                      torch.zeros_like(update1), update1)
                update2 = torch.where(update2 < clipping,
                                      torch.zeros_like(update2), update2)
                # Update the weights depending on the sign of the gradient
                f1 = self.f(x1 + update1)
                f2 = self.f(x2 + update2)
                # Update the weights depending on the sign of the gradient
                w1.data = torch.where(grad_w1 >= 0, f1, w1.data)
                w2.data = torch.where(grad_w2 >= 0, f2, w2.data)
                # Reset w2 to the maximum value and reset w1 the relative position of w2
                idx1 = w1 <= stop
                w1.data[idx1] = stop - (w2.data[idx1] - start)
                w2.data[idx1] = start
                # Reset w1 to the maximum value and reset w2 the relative position of w1
                idx2 = w2.data <= stop
                w2.data[idx2] = stop - (w1.data[idx2] - start)
                w1.data[idx2] = start
        return loss
