import torch
import torch.nn as nn
from modules import BaseEncoder, BaseDecoder, HyperEncoder, HyperDecoder, HyperCondDecoder
from distributions import NormalDistribution, FlexiblePrior
from utils import quantize
from torchvision.transforms.functional import resize


class SimpleModel(nn.Module):
    """
        I frame compression model
    """

    def __init__(
        self,
        input_dim,
        mid_dim,
        latent_dim,
        hyper_mid_dim,
        hyper_latent_dim,
        output_dim,
        activation="relu",
        vbr_dim=0,
        dec_add_latent=False,
    ):
        super().__init__()
        self.base_encoder = BaseEncoder(
            input_dim, mid_dim, latent_dim, activation=activation, vbr=vbr_dim
        )
        self.base_decoder = BaseDecoder(
            latent_dim * 2 if dec_add_latent else latent_dim,
            mid_dim,
            output_dim,
            activation="igdn" if activation == "gdn" else activation,
            vbr=vbr_dim,
        )
        self.prior = FlexiblePrior(channels=hyper_latent_dim)
        self.hyper_encoder = HyperEncoder(
            latent_dim, hyper_mid_dim, hyper_latent_dim, vbr=vbr_dim
        )
        self.hyper_decoder = HyperDecoder(
            hyper_latent_dim, hyper_mid_dim, latent_dim, vbr=vbr_dim
        )

    def encode(self, input, cond=None):
        latent = self.base_encoder(input, cond)
        hyper_latent = self.hyper_encoder(latent, cond)
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        latent_distribution = NormalDistribution(
            *self.hyper_decoder(q_hyper_latent, cond)
        )
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            'latent': latent,
            'hyper_latent': hyper_latent,
            'latent_distribution': latent_distribution
        }
        return q_latent, q_hyper_latent, state4bpp

    def decode(self, q_latent, additional_latent=None, cond=None):
        if additional_latent is not None:
            q_latent = torch.cat([q_latent, additional_latent], 1)
        return self.base_decoder(q_latent, cond)

    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        latent = state4bpp['latent']
        hyper_latent = state4bpp['hyper_latent']
        latent_distribution = state4bpp['latent_distribution']
        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(
                hyper_latent, "dequantize", self.prior.medians
            )
            q_latent = quantize(
                latent, "dequantize", latent_distribution.mean
            )
        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()
        bpp = (hyper_rate.sum() + cond_rate.sum()) / (B * H * W)
        resized_hyper_rate = resize(hyper_rate, size=(cond_rate.shape[-2], cond_rate.shape[-1]))
        resized_hyper_rate = resized_hyper_rate * (hyper_rate.shape[-1] * hyper_rate.shape[-2]) / (cond_rate.shape[-2] * cond_rate.shape[-1])
        return bpp, resized_hyper_rate.sum(1) + cond_rate.sum(1)

    def main_params(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "_medians" not in name:
                yield param

    def median_params(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "_medians" in name:
                yield param

    def extra_loss(self):
        return self.prior.get_extraloss()

    def forward(self, input, additional_latent=None, cond=None):
        q_latent, q_hyper_latent, state4bpp = self.encode(input, cond)
        output = self.decode(q_latent, additional_latent, cond)
        bpp, bpp_map = self.bpp(input.shape, state4bpp)
        # self.psnr = get_batch_psnr(recon, img, 1.)
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
            "bpp_map": bpp_map
        }


class CondModel(SimpleModel):
    def __init__(
        self,
        input_dim,
        mid_dim,
        latent_dim,
        hyper_mid_dim,
        hyper_latent_dim,
        output_dim,
        activation="relu",
        vbr_dim=0,
        dec_add_latent=False,
    ):
        super().__init__(
            input_dim,
            mid_dim,
            latent_dim,
            hyper_mid_dim,
            hyper_latent_dim,
            output_dim,
            activation,
            vbr_dim,
            dec_add_latent,
        )
        self.hyper_decoder = HyperCondDecoder(hyper_latent_dim, hyper_mid_dim, latent_dim, vbr=vbr_dim)

    def encode(self, input, additional_latent, additional_hyper_latent, cond=None):
        latent = self.base_encoder(input, cond)
        hyper_latent = self.hyper_encoder(latent, cond)
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        latent_distribution = NormalDistribution(
            *self.hyper_decoder(q_hyper_latent, additional_latent, additional_hyper_latent, cond)
        )
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            'latent': latent,
            'hyper_latent': hyper_latent,
            'latent_distribution': latent_distribution
        }
        return q_latent, q_hyper_latent, state4bpp

    def forward(self, input, additional_latent, additional_hyper_latent, cond=None):
        q_latent, q_hyper_latent, state4bpp = self.encode(input, additional_latent, additional_hyper_latent, cond)
        output = self.decode(q_latent, additional_latent, cond)
        bpp, bpp_map = self.bpp(input.shape, state4bpp)
        # self.psnr = get_batch_psnr(recon, img, 1.)
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
            "bpp_map": bpp_map
        }
