import torch
import kornia
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function


def get_min(tensor):  # :, C, H, W
    _min = tensor.view(tensor.size(0), tensor.size(1), -1).min(-1)[0].unsqueeze(-1).unsqueeze(-1)
    return _min


def get_max(tensor):  # :, C, H, W
    _max = tensor.view(tensor.size(0), tensor.size(1), -1).max(-1)[0].unsqueeze(-1).unsqueeze(-1)
    return _max


def get_batch_psnr(m1, m2, max_val=255):
    assert (m1.shape == m2.shape)
    N, C, H, W = m1.shape
    mse = ((m1 - m2)**2).view(N, -1).mean(1)
    return (10 * torch.log10((max_val**2) / mse)).mean()


def noise(input, scale):
    return input + scale*(torch.rand_like(input) - 0.5)


def round_w_offset(input, loc):
    diff = STERound.apply(input - loc)
    return diff + loc


def quantize(x, mode='noise', offset=None, scale=1):
    if mode == 'noise':
        return noise(x, scale)
    elif mode == 'round':
        return STERound.apply(x)
    elif mode == 'dequantize':
        return round_w_offset(x, offset)
    elif mode == 'scaled':
        return universal_quantize(x, offset, scale)
    else:
        raise NotImplementedError


def universal_quantize(x, m, s=1):
    return STERound.apply((x - m) / s) * s + m


def gaussian_blur(input, sigma):
    kernel_size = round(4 * sigma + 1)
    kernel = kornia.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma)).unsqueeze(0)
    return kornia.filter2D(input, kernel)


def gaussian_pyramids(input, base_sigma=1, m=5):
    output = [input]
    N, C, H, W = input.shape
    kernel = kornia.get_gaussian_kernel2d((5, 5), (base_sigma, base_sigma)).unsqueeze(0)
    for i in range(m):
        input = kornia.filter2D(input, kernel)
        if i == 0:
            output.append(input)
        else:
            tmp = input
            for j in range(i):
                tmp = F.interpolate(tmp, scale_factor=2., mode='bilinear', align_corners=True)
                # tmp = kornia.filter2D(tmp, kernel)
            output.append(tmp)
        input = F.interpolate(input, scale_factor=0.5)
    return torch.stack(output, 2)


def var_to_position(var, var_space):
    y_axis = np.linspace(-1, 1, len(var_space))
    x_axis = var_space
    y = []
    for i in range(len(var_space) - 1):
        slope = (y_axis[i + 1] - y_axis[i]) / (x_axis[i + 1] - x_axis[i])
        y.append(slope * (var - x_axis[i]) + y_axis[i])
    y, _ = torch.stack(y, -1).min(-1)
    return y.clamp(min=-1, max=1.)


class STERound(Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class UpperBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.min(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs <= b
        pass_through_2 = grad_output > 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None
