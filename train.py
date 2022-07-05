from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from data import load_data
from ssf import VideoCompressionModel
from utils import get_batch_psnr
import torch
import torch.nn.functional as F
import config
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='values from bash script')
parser.add_argument('--beta', type=float, required=True, help='beta value for single rate model')
parser.add_argument('--device', type=int, required=True, help='cuda device')
parser.add_argument('--mtype', type=str, required=True, help='mode type')
args = parser.parse_args()


def schedule_func(ep):
    return max(config.decay ** ep, config.minf)


# model and data config
model = VideoCompressionModel(
    config.filter_size,
    config.hyper_filter_size,
    True if "transform" in args.mtype else False,
    True if "cond" in args.mtype else False,
    vbr_dim=0
)
train_data, val_data = load_data(
    config.data_config, config.batch_size, pin_memory=False, num_workers=8
)
model_name = f'{config.data_config["dataset_name"]}-type_{args.mtype}-beta_{args.beta}-vbr_0'
if not os.path.isdir("./model_params"):
    os.mkdir("./model_params")
if config.model_checkpoint:
    loaded = torch.load(f"./model_params/{model_name}.pt", map_location=lambda storage, loc: storage)
    model.load_state_dict(loaded)
    print("model loaded")
model = model.to(args.device)


# training config
median_params = model.median_params()
optimizer = optim.Adam(
    [
        {"params": model.main_params(), "lr": config.lr},
        {"params": median_params[0], "lr": config.lr * 10},
        {"params": median_params[1], "lr": config.lr * 10},
        {"params": median_params[2], "lr": config.lr * 10}
    ]
)
scheduler = LambdaLR(optimizer, lr_lambda=[schedule_func for i in range(4)])
if config.opt_checkpoint:
    loaded = torch.load(f"./model_params/{model_name}-opt.pt", map_location=lambda storage, loc: storage)
    optimizer.load_state_dict(loaded)
    print("optimizer loaded")


def train(batch, beta):
    model.train()
    optimizer.zero_grad()
    batch = batch.to(args.device)
    state = model(batch)
    mse = F.mse_loss(state["output"], batch)
    bpp = state["bpp"].mean()
    loss = beta * bpp + mse
    loss.backward()
    median_losses = model.compute_extra_loss()
    for mloss in median_losses:
        mloss.backward()
    torch.nn.utils.clip_grad_norm_(model.main_params(), config.grad_clip)
    optimizer.step()
    with torch.no_grad():
        psnr = [get_batch_psnr(state["output"][i].clamp(0, 1), batch[i], 1.0) for i in range(batch.shape[0])]
        state["psnr"] = torch.stack(psnr, 0)
        state["loss"] = loss
        state["mse"] = mse
        return state


def test(batch):
    model.eval()
    with torch.no_grad():
        batch = batch.to(args.device)
        state = model(batch)
        psnr = [get_batch_psnr(state["output"][i].clamp(0, 1), batch[i], 1.0) for i in range(batch.shape[0])]
        state["psnr"] = torch.stack(psnr, 0)
        return state


finish = False
if os.path.isdir(f'{config.tbdir}/{args.mtype}/{model_name}'):
    shutil.rmtree(f'{config.tbdir}/{args.mtype}/{model_name}')
writer = SummaryWriter(f"{config.tbdir}/{args.mtype}/{model_name}")
step = 0
while True:

    for train_batch_idx, train_batch in enumerate(train_data):

        state = train(train_batch, args.beta)
        if step % config.log_checkpoint_step == 0:
            writer.add_images(
                "train/recon_sample", state["output"][:, 0, ...].clamp(0, 1), step // config.log_checkpoint_step
            )
            writer.add_images(
                "train/warped_sample",
                state["warped"][:, 0, ...],
                step // config.log_checkpoint_step,
            )
            writer.add_scalar("train/loss", state["loss"], step // config.log_checkpoint_step)
            writer.add_scalar("train/mse", state["mse"], step // config.log_checkpoint_step)
            writer.add_scalar("train/bpp", state["bpp"].mean(), step // config.log_checkpoint_step)
            writer.add_text(
                "train/bpp_seq_per_batch",
                "--".join("%.4f" % e for e in state["bpp"].tolist()),
                step // config.log_checkpoint_step,
            )
            writer.add_text(
                "train/psnr_seq_per_batch",
                "--".join("%.3f" % e for e in state["psnr"].tolist()),
                step // config.log_checkpoint_step,
            )

            # validation
            num_of_sample = 0
            bpp_seq = 0
            psnr_seq = 0
            bpp = 0
            for test_batch_idx, test_batch in enumerate(val_data):
                state = test(test_batch)
                if test_batch_idx == 0:
                    writer.add_images(
                        "val/recon_sample",
                        state["output"][:, 0, ...].clamp(0, 1),
                        step // config.log_checkpoint_step,
                    )
                    writer.add_images(
                        "val/warped_sample",
                        state["warped"][:, 0, ...],
                        step // config.log_checkpoint_step,
                    )
                num_of_sample += test_batch.shape[1]
                bpp_seq += state["bpp"] * test_batch.shape[1]
                psnr_seq += state["psnr"] * test_batch.shape[1]
                bpp += state["bpp"].mean() * test_batch.shape[1]
            writer.add_scalar("val/bpp", bpp / num_of_sample, step // config.log_checkpoint_step)
            writer.add_text(
                "val/bpp_seq_per_batch",
                "--".join("%.4f" % e for e in (bpp_seq / num_of_sample).tolist()),
                step // config.log_checkpoint_step,
            )
            writer.add_text(
                "val/psnr_seq_per_batch",
                "--".join("%.3f" % e for e in (psnr_seq / num_of_sample).tolist()),
                step // config.log_checkpoint_step,
            )
        step += 1

        if step % config.save_step == 0:
            torch.save(model.state_dict(), f"./model_params/{model_name}.pt")
            torch.save(optimizer.state_dict(), f"./model_params/{model_name}-opt.pt")

        if (step % config.scheduler_step == 0) and (step != 0):
            scheduler.step()

        if step > config.n_step:
            finish = True
            break

    if finish:
        break

torch.save(model.state_dict(), f"./model_params/{model_name}.pt")
torch.save(optimizer.state_dict(), f"./model_params/{model_name}-opt.pt")
