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
        if not 0.0 <= scale:
            raise ValueError("Invalid scale value: {}".format(scale))
        self.set_field(field, scale)
        defaults = dict(lr=lr,
                        scale=scale,
                        eps=eps,
                        field=field)
        super(Magnetoionic, self).__init__(params, defaults)

    def set_field(self, field, scale):
        # All functions are scaled between -1 and 1 to have both positive and negative weights
        if field == "simple-strong":
            # Set of functions between -1 and 1

            def f_strong_minus(x): return -24.77 * \
                torch.log(0.287*torch.log(x+128))+9.11

            def f_strong_inverse_minus(y): return torch.exp(
                torch.exp((y - 9.11)/-24.77)/0.287) - 128

            def f_strong_plus(x): return 0.443 * \
                torch.log(2.53*torch.log(x+1.66)) - 0.11

            def f_strong_inverse_plus(y): return torch.exp(
                torch.exp((y + 0.11)/0.443)/2.53) - 1.66

            self.f_minus = lambda x: 2*scale * f_strong_minus(x) - scale
            self.f_inverse_minus = lambda y: f_strong_inverse_minus(
                (y + scale)/(2*scale))

            self.f_plus = lambda x: 2*scale * f_strong_plus(x) - scale
            self.f_inverse_plus = lambda y: f_strong_inverse_plus(
                (y + scale)/(2*scale))

        elif field == "simple-weak":
            # Set of functions between -1 and 1

            def f_weak_minus(x): return -1.28 * \
                torch.log(0.997*torch.log(x+5.02))+1.583

            def f_weak_inverse_minus(y): return torch.exp(
                torch.exp((y - 1.583)/-1.28)/0.997) - 5.02

            def f_weak_plus(x): return 0.9066 * \
                torch.log(1.176*torch.log(x+4.42))-0.507

            def f_weak_inverse_plus(y): return torch.exp(
                torch.exp((y + 0.507)/0.9066)/1.176) - 4.42

            self.f_minus = lambda x: 2*scale * f_weak_minus(x) - scale
            self.f_inverse_minus = lambda y: f_weak_inverse_minus(
                (y + scale)/(2*scale))

            self.f_plus = lambda x: 2*scale * f_weak_plus(x) - scale
            self.f_inverse_plus = lambda y: f_weak_inverse_plus(
                (y + scale)/(2*scale))

        elif field == "simple-linear":
            self.f_minus = lambda x: (-2*(1/27)*x + 1)*scale
            self.f_inverse_minus = lambda y: -(y/scale - 1)/(2*(1/27))

            self.f_plus = lambda x: (2*(1/27)*x - 1)*scale
            self.f_inverse_plus = lambda y: (y/scale + 1)/(2*(1/27))

        elif field == "double-linear":
            self.f = lambda x: (-2*(1/27)*x + 1)*scale
            self.f_inv = lambda y: -(y/scale - 1)/(2*(1/27))

        elif field == "double-exponential":
            def f_weak_minus(x): return -1.28 * \
                torch.log(0.997*torch.log(x+5.02))+1.583

            def f_weak_inverse_minus(y): return torch.exp(
                torch.exp((y - 1.583)/-1.28)/0.997) - 5.02

            self.f = lambda x: 2*scale * f_weak_minus(x) - scale
            self.f_inv = lambda y: f_weak_inverse_minus(
                (y + scale)/(2*scale))
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
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                lr = group['lr']
                scale = group['scale']
                field = group['field']
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                state['step'] += 1
                init = 0.01

                if "double" in field and state['step'] == 1:
                    state[f'w1_{i}'] = torch.ones_like(
                        p.data) - init
                    state[f'w2_{i}'] = torch.ones_like(
                        p.data) - init - torch.distributions.Uniform(-init, init).sample(p.data.shape).to(p.data.device)
                # When we do simple, we have only one device so we project the current weight on the functions depending on the sign of the gradient
                if "simple" in self.field:
                    # Compute x = f^-1(w), the previous x of the old weights
                    x = torch.where(grad >= 0,
                                    self.f_inverse_plus(p.data),
                                    self.f_inverse_minus(p.data))
                    # Compute the value of the polynom at the new position to get the new weights w_t = x_{t-1} + lr * grad
                    w_t = torch.where(grad >= 0,
                                      self.f_plus(x - lr*torch.abs(grad)),
                                      self.f_minus(x - lr*torch.abs(grad)))
                    w_t = torch.clamp(w_t, -scale, scale)
                # With two devices, we need two matrices to store the weights w1 and w2
                elif "double" in self.field:
                    w1 = state[f'w1_{i}']
                    w2 = state[f'w2_{i}']
                    # Retrieve the number of pulses associated with the current weight state
                    x1 = self.f_inv(w1)
                    x2 = self.f_inv(w2)
                    # Compute the new value of the weights given a pulse set as the gradient
                    f1 = self.f(x1 + lr*torch.abs(grad))
                    f2 = self.f(x2 + lr*torch.abs(grad))
                    # Update w1 only when the gradient is negative
                    w1 = torch.where(grad < 0,
                                     f1,
                                     w1)
                    # Ipdate w2 only when the gradient is positive
                    w2 = torch.where(grad >= 0,
                                     f2,
                                     w2)
                    # Here, we don't clamp, as we work with difference, we can simply reset the distance
                    idx = w1 < -scale
                    temp = w2[idx]
                    w2[idx] = scale
                    w1[idx] = w1[idx] + scale - temp
                    idx = w2 < -scale
                    temp = w1[idx]
                    w1[idx] = scale
                    w2[idx] = w2[idx] + scale - temp
                    # Update the state
                    state[f'w1_{i}'] = w1
                    state[f'w2_{i}'] = w2
                    w_t = w2-w1
                p.data = w_t
        return loss
