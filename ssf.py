import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import gaussian_pyramids
from models import SimpleModel, CondModel


class VideoCompressionModel(nn.Module):
    """
        P frame model with additional I frame model
    """

    def __init__(
        self,
        dims,
        hyper_dims,
        use_scaler=False,
        use_cond_residual=False,
        gaussian_dim=5,
        base_scale=1.0,
        activation="relu",
        vbr_dim=0,
    ):
        super().__init__()
        self.iframe_model = SimpleModel(
            3, dims, hyper_dims, hyper_dims, hyper_dims, 3, activation, int(vbr_dim)
        )
        self.flow_model = SimpleModel(
            6,
            dims,
            hyper_dims,
            hyper_dims,
            hyper_dims,
            6 if use_scaler else 3,
            activation,
            int(vbr_dim),
        )
        if use_cond_residual:
            self.residual_model = CondModel(
                3,
                dims,
                hyper_dims,
                hyper_dims,
                hyper_dims,
                3,
                activation,
                int(vbr_dim),
                dec_add_latent=True,
            )
        else:
            self.residual_model = SimpleModel(
                3,
                dims,
                hyper_dims,
                hyper_dims,
                hyper_dims,
                3,
                activation,
                int(vbr_dim),
                dec_add_latent=True,
            )
        self.gaussian_dim = int(gaussian_dim)
        self.base_scale = base_scale
        self.use_scaler = use_scaler
        self.use_cond_residual = use_cond_residual

    def main_params(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "_medians" not in name:
                yield param

    def median_params(self, recurse=True):
        params = []
        for name, param in self.named_parameters(recurse=recurse):
            if "_medians" in name:
                params.append(param)
        return params

    def scale_space_warp(self, input, flow):
        # predict the corresponding sigma and then convert to the scale space location
        N, C, H, W = input.shape
        assert flow.shape == (N, 3, H, W)
        flow = flow.unsqueeze(0)
        multi_scale = gaussian_pyramids(input, self.base_scale, self.gaussian_dim)
        h = torch.arange(H, device=input.device, dtype=input.dtype)
        w = torch.arange(W, device=input.device, dtype=input.dtype)
        d = torch.zeros(1, device=input.device, dtype=input.dtype)
        grid = torch.stack(torch.meshgrid(d, h, w)[::-1], -1).unsqueeze(0)
        grid = grid.expand(N, -1, -1, -1, -1)
        flow = flow.permute(1, 0, 3, 4, 2)  # N, 1, H, W, 3

        # reparameterization
        # var_channel = (flow[..., -1].exp())**2
        # var_space = [0.] + [(2.**i * self.base_scale)**2 for i in range(self.gaussian_dim)]
        # d_offset = var_to_position(var_channel, var_space).unsqueeze(-1)
        d_offset = flow[..., -1].clamp(min=-1.0, max=1.0).unsqueeze(-1)

        flow = torch.cat((flow[..., :2], d_offset), -1)
        flow_grid = flow + grid
        flow_grid[..., 0] = 2.0 * flow_grid[..., 0] / max(W - 1.0, 1.0) - 1.0
        flow_grid[..., 1] = 2.0 * flow_grid[..., 1] / max(H - 1.0, 1.0) - 1.0
        warped = F.grid_sample(
            multi_scale, flow_grid, padding_mode="border", align_corners=True
        ).squeeze(2)
        return warped

    def compute_extra_loss(self):
        return (
            self.iframe_model.extra_loss(),
            self.flow_model.extra_loss(),
            self.residual_model.extra_loss(),
        )

    def iframe_forward(self, frame, cond=None):
        state = self.iframe_model(frame, cond=cond)
        return state

    def pframe_forward(self, cur_frame, prev_frame, cond=None):
        flow_input = torch.cat([cur_frame, prev_frame], 1)
        flow_state = self.flow_model(flow_input, cond=cond)
        if self.use_scaler:
            flow = flow_state["output"][:, :3, ...]
            scale = flow_state["output"][:, 3:, ...].exp()
        else:
            flow = flow_state["output"]
            scale = 1
        warped = self.scale_space_warp(prev_frame, flow)
        residual = (cur_frame - warped) / scale
        if self.use_cond_residual:
            residual_state = self.residual_model(
                residual, flow_state["q_latent"], flow_state["q_hyper_latent"], cond
            )
        else:
            residual_state = self.residual_model(residual, flow_state["q_latent"], cond)
        output = warped + residual_state["output"] * scale
        bpp = flow_state["bpp"] + residual_state["bpp"]
        return {"output": output, "bpp": bpp, "warped": warped.clamp(0, 1), "bpp_map": flow_state["bpp_map"] + residual_state["bpp_map"]}

    def forward(self, video, cond=None):
        T = video.shape[0]
        states, prev_frame = [], None
        for t in range(T):
            if t == 0:
                state = self.iframe_forward(video[0], cond=cond)
                prev_frame = state["output"].detach()
            else:
                state = self.pframe_forward(video[t], prev_frame.clamp(0, 1), cond=cond)
                prev_frame = state["output"]
            states.append(state)
        return {
            "output": torch.stack([s["output"] for s in states], 0),
            "bpp": torch.stack([s["bpp"] for s in states], 0),
            "warped": torch.stack([s["warped"] for s in states[1:]], 0)
        }
