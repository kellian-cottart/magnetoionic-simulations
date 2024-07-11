import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Magnetoionic(torch.optim.Optimizer):
    """ Magnetoionic Optimizer made for simulating magnetic fields in magnetoinic devices

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        field (str, optional): type of magnetic field applied, either "strong", "weak" or "linear"
        scale (float, optional): scale of the functions f_minus and f_plus
        eps (float, optional): term added to the denominator to improve numerical stability
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 field="weak",
                 scale=1,
                 eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if field not in ["strong", "weak", "linear"]:
            raise ValueError(
                "Invalid field type: {}".format(field))
        if not 0.0 <= scale:
            raise ValueError("Invalid scale value: {}".format(scale))
        self.set_field(field, scale)
        defaults = dict(lr=lr,
                        scale=scale,
                        eps=eps)
        super(Magnetoionic, self).__init__(params, defaults)

        # Depending on the field applied, we have different functions
        # Functions are rescale according to (2f(x) - 1)*scale

    def __setstate__(self, state):
        super(Magnetoionic, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def set_field(self, field, scale):
        if field == "strong":
            self.f_minus = lambda x: (-2*(1/27)*x + 1)*scale
            self.f_inverse_minus = lambda y: -(y/scale - 1)/(2*(1/27))

            self.f_plus = lambda x: (1 - 2*torch.exp(-x/11))*scale
            self.f_inverse_plus = lambda y: -11 * torch.log((1-y/scale)/2)
        elif field == "weak":
            self.f_minus = lambda x: (2.16*torch.exp(-x/11)-1.16)*scale
            self.f_inverse_minus = lambda y: - \
                11 * torch.log((1.16+y/scale)/2.16)

            self.f_plus = lambda x: (1 - 2*torch.exp(-x/18))*scale
            self.f_inverse_plus = lambda y: -18 * torch.log((1-y/scale)/2)
        elif field == "linear":
            self.f_minus = lambda x: -x
            self.f_inverse_minus = lambda y: -y

            self.f_plus = lambda x: x
            self.f_inverse_plus = lambda y: y

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
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                lr = group['lr']
                scale = group['scale']
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                state['step'] += 1
                # Compute x = f^-1(w), the previous x of the old weights
                x = torch.where(grad >= 0,
                                self.f_inverse_plus(p.data),
                                self.f_inverse_minus(p.data))
                # Compute the value of the polynom at the new position to get the new weights w_t = x_{t-1} + lr * grad
                w_t = torch.where(grad >= 0,
                                  self.f_plus(x - lr*torch.abs(grad)),
                                  self.f_minus(x - lr*torch.abs(grad)))
                w_t = torch.clamp(w_t, -scale, scale)
                # Update the parameters
                p.data = w_t
        return loss
