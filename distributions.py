import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Uniform
from utils import LowerBound
from modules import ConvLSTMCell

class NormalDistribution:
    '''
        A normal distribution
    '''
    def __init__(self, loc, scale):
        assert loc.shape == scale.shape
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        return self.loc.detach()

    def std_cdf(self, inputs):
        half = 0.5
        const = -(2**-0.5)
        return half * torch.erfc(const * inputs)

    def sample(self):
        return self.scale * torch.randn_like(self.scale) + self.loc

    def likelihood(self, x, min=1e-9):
        x = torch.abs(x - self.loc)
        upper = self.std_cdf((.5 - x) / self.scale)
        lower = self.std_cdf((-.5 - x) / self.scale)
        return LowerBound.apply(upper - lower, min)

    def scaled_likelihood(self, x, s=1, min=1e-9):
        x = torch.abs(x - self.loc)
        s = s * .5
        upper = self.std_cdf((s - x) / self.scale)
        lower = self.std_cdf((-s - x) / self.scale)
        return LowerBound.apply(upper - lower, min)


class PriorFunction(nn.Module):
    #  A Custom Function described in Balle et al 2018. https://arxiv.org/pdf/1802.01436.pdf
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, parallel_dims, in_features, out_features, scale, bias=True):
        super(PriorFunction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.5, 0.5)

    def forward(self, input, detach=False):
        # input shape (channel, batch_size, in_features)
        if detach:
            return torch.matmul(input, F.softplus(self.weight.detach())) + self.bias.detach()
        return torch.matmul(input, F.softplus(self.weight)) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias
                                                                 is not None)


class FlexiblePrior(nn.Module):
    '''
        A prior model described in Balle et al 2018 Appendix 6.1 https://arxiv.org/pdf/1802.01436.pdf
        return the boxshape likelihood
    '''
    def __init__(self, channels=256, dims=[3, 3, 3], init_scale=10.):
        super(FlexiblePrior, self).__init__()
        dims = [1] + dims + [1]
        self.chain_len = len(dims) - 1
        scale = init_scale**(1 / self.chain_len)
        h_b = []
        for i in range(self.chain_len):
            init = np.log(np.expm1(1 / scale / dims[i + 1]))
            h_b.append(PriorFunction(channels, dims[i], dims[i + 1], init))
        self.affine = nn.ModuleList(h_b)
        self.a = nn.ParameterList(
            [nn.Parameter(torch.zeros(channels, 1, 1, 1, dims[i + 1])) for i in range(self.chain_len - 1)])

        # optimize the medians to fix the offset issue
        self._medians = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @property
    def medians(self):
        return self._medians.detach()

    def cdf(self, x, logits=True, detach=False):
        x = x.transpose(0, 1).unsqueeze(-1)  # C, N, H, W, 1
        if detach:
            for i in range(self.chain_len - 1):
                x = self.affine[i](x, detach)
                x += torch.tanh(self.a[i].detach()) * torch.tanh(x)
            if logits:
                return self.affine[-1](x, detach).squeeze(-1).transpose(0, 1)
            return torch.sigmoid(self.affine[-1](x, detach)).squeeze(-1).transpose(0, 1)

        # not detached
        for i in range(self.chain_len - 1):
            x = self.affine[i](x)
            x += torch.tanh(self.a[i]) * torch.tanh(x)
        if logits:
            return self.affine[-1](x).squeeze(-1).transpose(0, 1)
        return torch.sigmoid(self.affine[-1](x)).squeeze(-1).transpose(0, 1)

    def pdf(self, x):
        cdf = self.cdf(x, False)
        jac = torch.ones_like(cdf)
        pdf = torch.autograd.grad(cdf, x, grad_outputs=jac)[0]
        return pdf

    def get_extraloss(self):
        target = 0
        logits = self.cdf(self._medians, detach=True)
        extra_loss = torch.abs(logits - target).sum()
        return extra_loss

    def likelihood(self, x, min=1e-9):
        lower = self.cdf(x - 0.5, True)
        upper = self.cdf(x + 0.5, True)
        sign = -torch.sign(lower + upper).detach()
        upper = torch.sigmoid(upper * sign)
        lower = torch.sigmoid(lower * sign)
        return LowerBound.apply(torch.abs(upper - lower), min)

    def scaled_likelihood(self, x, s, min=1e-9):
        lower = self.cdf(x - 0.5 * s, True)
        upper = self.cdf(x + 0.5 * s, True)
        sign = -torch.sign(lower + upper).detach()
        upper = torch.sigmoid(upper * sign)
        lower = torch.sigmoid(lower * sign)
        return LowerBound.apply(torch.abs(upper - lower), min)

    def icdf(self, xi, method='bisection', max_iterations=1000, tol=1e-9, **kwargs):
        if method == 'bisection':
            init_interval = [-1, 1]
            left_endpoints = torch.ones_like(xi) * init_interval[0]
            right_endpoints = torch.ones_like(xi) * init_interval[1]

            def f(z):
                return self.cdf(z, logits=False, detach=True) - xi

            while True:
                if (f(left_endpoints) < 0).all():
                    break
                else:
                    left_endpoints = left_endpoints * 2
            while True:
                if (f(right_endpoints) > 0).all():
                    break
                else:
                    right_endpoints = right_endpoints * 2

            for i in range(max_iterations):
                mid_pts = 0.5 * (left_endpoints + right_endpoints)
                mid_vals = f(mid_pts)
                pos = mid_vals > 0
                non_pos = torch.logical_not(pos)
                neg = mid_vals < 0
                non_neg = torch.logical_not(neg)
                left_endpoints = left_endpoints * non_neg.float() + mid_pts * neg.float()
                right_endpoints = right_endpoints * non_pos.float() + mid_pts * pos.float()
                if (torch.logical_and(non_pos, non_neg)).all() or torch.min(right_endpoints - left_endpoints) <= tol:
                    print(f'bisection terminated after {i} its')
                    break

            return mid_pts
        else:
            raise NotImplementedError

    def sample(self, img, shape):
        uni = torch.rand(shape, device=img.device)
        return self.icdf(uni)


class SpatialAutoregressivePrior(nn.Module):
    '''
        Spatial autoregressive prior
    '''
    def __init__(self, filters=(1, 3, 1), inter_z_size=512, z_size=256, dims=[3, 3, 3], scale=10., bound=1e-9):
        super(SpatialAutoregressivePrior, self).__init__()
        self.init_prior = FlexiblePrior(z_size, dims, scale, bound)
        conv_stack = []
        for i in range(len(filters)):
            if i == 0:
                conv_stack.append(nn.ConvTranspose2d(z_size, inter_z_size, filters[i]))
            else:
                conv_stack.append(nn.ConvTranspose2d(inter_z_size, inter_z_size, filters[i]))
            conv_stack.append(nn.LeakyReLU(0.2, True))
        self.conv = nn.Sequential(*conv_stack)
        self.loc_conv = nn.ConvTranspose2d(inter_z_size, z_size, 1)
        self.scale_conv = nn.ConvTranspose2d(inter_z_size, z_size, 1)

    def kl(self, x):
        def get_ring(x):
            transposed = False
            if x.shape[2] > x.shape[3]:
                x = x.transpose(2, 3)
                transposed = True
            N, C, H, W = x.shape
            a = x[:, :, 0, :]
            c = x[:, :, H - 1, :]
            if H == 1:
                ring = a
            elif H == 2:
                ring = torch.cat([a, c.flip(-1)], -1)
            else:
                b = x[:, :, 1:-1, -1]
                d = x[:, :, 1:-1, 0].flip(-1)
                ring = torch.cat([a, b, c.flip(-1), d], -1)
            if transposed:
                ring = ring.flip(-1).roll(1, -1)
            return ring

        kl, dist = 0, None
        N, C, H, W = x.shape
        if H % 2 == 0:
            tanchor = H // 2 - 1
            banchor = H // 2 + 1
        else:
            tanchor = H // 2
            banchor = H // 2 + 1
        lanchor = tanchor
        ranchor = W - tanchor
        while tanchor >= 0:
            y = x[:, :, tanchor:banchor, lanchor:ranchor]
            if dist is None:
                kl += -self.init_prior(y).log2().reshape(N, -1).sum(1)
            else:
                ring = get_ring(y)
                kl += -dist.likelihood(ring).log2().reshape(N, -1).sum(1)
            h = self.conv(y)
            mu = self.loc_conv(h)
            sigma = self.scale_conv(h)
            mu = get_ring(mu)
            sigma = get_ring(sigma).exp()
            dist = NormalExt(mu, sigma)
            tanchor -= 1
            banchor += 1
            lanchor -= 1
            ranchor += 1
        return kl


class LSTMPrior(nn.Module):
    '''
        Auto-regressive prior modeled by a ConvLSTM
    '''
    def __init__(self, filter_size, rnn_hidden_size):
        super(LSTMPrior, self).__init__()
        self.lstm = ConvLSTMCell(filter_size, rnn_hidden_size, 3)
        self.loc_conv = nn.Conv2d(rnn_hidden_size, filter_size, 3, 1, 1)
        self.scale_conv = nn.Conv2d(rnn_hidden_size, filter_size, 3, 1, 1)

    def forward(self, h, hidden=None):
        if hidden is None:
            h, c = self.lstm(h)
        else:
            h, c = self.lstm(h, hidden)
        loc = self.loc_conv(h)
        scale = self.scale_conv(h).exp()
        return NormalExt(loc, scale), (h, c)


class MarkovPrior(nn.Module):
    '''
        markov autoregressive prior
    '''
    def __init__(self, filter_size, hyper_filter_size, img_c):
        super(MarkovPrior, self).__init__()
        self.loc_conv = nn.Sequential(nn.Conv2d(img_c, filter_size, 4, 2, 1), nn.ReLU(True),
                                      nn.Conv2d(filter_size, filter_size, 4, 2, 1), nn.ReLU(True),
                                      nn.Conv2d(filter_size, filter_size, 4, 2, 1), nn.ReLU(True),
                                      nn.Conv2d(filter_size, hyper_filter_size, 4, 2, 1))
        self.scale_conv = nn.Sequential(nn.Conv2d(img_c, filter_size, 4, 2, 1), nn.ReLU(True),
                                        nn.Conv2d(filter_size, filter_size, 4, 2, 1), nn.ReLU(True),
                                        nn.Conv2d(filter_size, filter_size, 4, 2, 1), nn.ReLU(True),
                                        nn.Conv2d(filter_size, hyper_filter_size, 4, 2, 1))

    def forward(self, x):
        loc = self.loc_conv(x)
        scale = LowerBound.apply(self.scale_conv(x).exp(), 0.11)
        return NormalExt(loc, scale)


class Posterior(nn.Module):
    '''
        Posterior for P frames
    '''
    def __init__(self, filter_size=None):
        super(Posterior, self).__init__()

    def forward(self, h):
        return Uniform(h - 0.5, h + 0.5)


class InitPosterior(nn.Module):
    '''
        Posterior for I frames
    '''
    def __init__(self, img_filter_size=None):
        super(InitPosterior, self).__init__()
        # self.loc_conv = nn.Conv2d(img_filter_size, img_filter_size, 3, 1, 1)

    def forward(self, loc):
        # loc = self.loc_conv(loc)
        return Uniform(loc - 0.5, loc + 0.5)
