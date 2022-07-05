import torch.nn as nn
import torch
import inspect
from utils import LowerBound


class VBRCondition(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.scale = nn.Linear(input_dim, output_dim)
        self.shift = nn.Linear(input_dim, output_dim)

    def forward(self, input, cond):
        scale = self.scale(cond)
        shift = self.shift(cond)
        return input * scale + shift


class CustomSequential(nn.Sequential):
    def forward(self, input, cond):
        for module in self:
            if "cond" in str(inspect.signature(module.forward)):
                input = module(input, cond)
            else:
                input = module(input)
        return input


class Coder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = CustomSequential()

    def build_activation(self, act, param=None):
        if act == "relu":
            return nn.ReLU(True)
        elif act == "gdn":
            return GDN1(param)
        elif act == "igdn":
            return GDN1(param, True)
        elif act == "leakyrelu":
            return nn.LeakyReLU(0.2, inplace=True)
        else:
            raise NotImplementedError

    def forward(self, input, cond=None):
        input = self.network(input, cond)
        return input


class BaseEncoder(Coder):
    """
        Base encoder
    """

    def __init__(self, dim_in, dim_mid, dim_out, num_of_layer=4, activation="relu", vbr=0):
        super().__init__()
        for i in range(num_of_layer):
            if i == 0:
                self.network.add_module(f"conv_{i}", nn.Conv2d(dim_in, dim_mid, 5, 2, 2))
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))
            elif i != (num_of_layer - 1):
                self.network.add_module(f"conv_{i}", nn.Conv2d(dim_mid, dim_mid, 5, 2, 2))
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))
            else:
                self.network.add_module(f"conv_{i}", nn.Conv2d(dim_mid, dim_out, 5, 2, 2))


class BaseDecoder(Coder):
    """
        Base decoder
    """

    def __init__(self, dim_in, dim_mid, dim_out, num_of_layer=4, activation="relu", vbr=0):
        super().__init__()
        for i in range(num_of_layer):
            if i == 0:
                self.network.add_module(
                    f"deconv_{i}", nn.ConvTranspose2d(dim_in, dim_mid, 5, 2, 2, 1)
                )
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))
            elif i != (num_of_layer - 1):
                self.network.add_module(
                    f"deconv_{i}", nn.ConvTranspose2d(dim_mid, dim_mid, 5, 2, 2, 1)
                )
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))
            else:
                self.network.add_module(
                    f"deconv_{i}", nn.ConvTranspose2d(dim_mid, dim_out, 5, 2, 2, 1)
                )


class HyperEncoder(Coder):
    """
        Hyper encoder
    """

    def __init__(self, dim_in, dim_mid, dim_out, num_of_layer=3, activation="relu", vbr=0):
        super().__init__()
        for i in range(num_of_layer):
            if i == 0:
                self.network.add_module(f"conv_{i}", nn.Conv2d(dim_in, dim_mid, 3, 1, 1))
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))
            elif i != (num_of_layer - 1):
                self.network.add_module(f"conv_{i}", nn.Conv2d(dim_mid, dim_mid, 5, 2, 2))
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))
            else:
                self.network.add_module(f"conv_{i}", nn.Conv2d(dim_mid, dim_out, 5, 2, 2))


class HyperDecoder(Coder):
    """
        Hyper decoder
    """

    def __init__(self, dim_in, dim_mid, dim_out, num_of_layer=3, activation="relu", vbr=0):
        super().__init__()
        for i in range(num_of_layer):
            if i == 0:
                self.network.add_module(
                    f"deconv_{i}", nn.ConvTranspose2d(dim_in, dim_mid, 5, 2, 2, 1),
                )
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))
            elif i != (num_of_layer - 1):
                self.network.add_module(
                    f"deconv_{i}", nn.ConvTranspose2d(dim_mid, dim_mid, 5, 2, 2, 1),
                )
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))
            else:
                self.network.add_module(
                    f"deconv_{i}", nn.Conv2d(dim_mid, dim_out * 2, 3, 1, 1),
                )

    def forward(self, hyper_latent, cond=None):
        output = self.network(hyper_latent, cond)
        mean = output[:, : output.shape[1] // 2]
        scale = output[:, output.shape[1] // 2 :]
        return mean, LowerBound.apply(scale, 0.11)


class HyperCondDecoder(Coder):
    """
        Hyper conditional decoder
    """

    def __init__(
        self,
        dim_in,
        dim_mid,
        dim_out,
        num_of_upper_layer=2,
        num_of_lower_layer=2,
        activation="relu",
        vbr=0,
    ):
        super().__init__()
        for i in range(num_of_upper_layer):
            if i == 0:
                self.network.add_module(
                    f"upper_deconv_{i}", nn.ConvTranspose2d(dim_in * 2, dim_mid, 5, 2, 2, 1),
                )
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))
            else:
                self.network.add_module(
                    f"upper_deconv_{i}", nn.ConvTranspose2d(dim_mid, dim_mid, 5, 2, 2, 1),
                )
                if vbr > 0:
                    self.network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.network.add_module(f"act_{i}", self.build_activation(activation, dim_mid))

        self.lower_network = CustomSequential()
        for i in range(num_of_lower_layer):
            if i == 0:
                self.lower_network.add_module(
                    f"lower_deconv_{i}", nn.Conv2d((dim_mid + dim_out), dim_mid, 3, 1, 1),
                )
                if vbr > 0:
                    self.lower_network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.lower_network.add_module(
                    f"act_{i}", self.build_activation(activation, dim_mid)
                )
            elif i != (num_of_lower_layer - 1):
                self.lower_network.add_module(
                    f"lower_deconv_{i}", nn.Conv2d(dim_mid, dim_mid, 3, 1, 1),
                )
                if vbr > 0:
                    self.lower_network.add_module(f"vbr_{i}", VBRCondition(vbr, dim_mid))
                self.lower_network.add_module(
                    f"act_{i}", self.build_activation(activation, dim_mid)
                )
            else:
                self.lower_network.add_module(
                    f"lower_deconv_{i}", nn.Conv2d(dim_mid, dim_out * 2, 3, 1, 1),
                )

    def forward(self, hyper_latent, flow_latent, flow_hyper_latent, cond=None):
        stacked_hyper_latent = torch.cat([hyper_latent, flow_hyper_latent], 1)
        stacked_hyper_latent = self.network(stacked_hyper_latent, cond)

        stacked_hyper_latent = torch.cat([stacked_hyper_latent, flow_latent], 1,)
        stacked_hyper_latent = self.lower_network(stacked_hyper_latent, cond)

        mean = stacked_hyper_latent[:, : stacked_hyper_latent.shape[1] // 2]
        scale = stacked_hyper_latent[:, stacked_hyper_latent.shape[1] // 2 :]

        return mean, LowerBound.apply(scale, 0.11)


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self, ch, inverse=False, beta_min=1e-5, gamma_init=0.1, reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** 0.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class GDN1(GDN):
    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(torch.abs(inputs), gamma, beta)
        # norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state=None):
        N, C, H, W = input_tensor.shape
        if cur_state is None:
            h_cur, c_cur = self.init_hidden(N, (H, W))
        else:
            h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size, self.hidden_dim, height, width, device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size, self.hidden_dim, height, width, device=self.conv.weight.device,
            ),
        )
