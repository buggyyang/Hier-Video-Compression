import torch
import os
import numpy as np
import config
from PIL import Image
import torchvision.transforms.functional as VF
from ssf import VideoCompressionModel
from utils import get_batch_psnr
import argparse


def frame_processing(img, conv_factor=128):
    # img = VF.center_crop(img, 256)
    W, H = img.size
    W_residual, H_residual = 0, 0
    W_residual = (W % conv_factor) // 2
    H_residual = (H % conv_factor) // 2
    img = img.crop((W_residual, H_residual, W - W_residual, H - H_residual))
    tensor = VF.to_tensor(img)

    return tensor.unsqueeze(0)


parser = argparse.ArgumentParser(description='values from bash script')
parser.add_argument('--device', type=int, required=True, help='cuda device')
parser.add_argument('--mtype', type=str, required=True, help='mode type')
args = parser.parse_args()

# model and data config
model = VideoCompressionModel(
    config.filter_size,
    config.hyper_filter_size,
    True if "transform" in args.mtype else False,
    True if "cond" in args.mtype else False,
    vbr_dim=0,
)
bpp_table = np.zeros((len(config.betas), len(config.test_videos)))
psnr_table = np.zeros((len(config.betas), len(config.test_videos)))

for bi, b in enumerate(config.betas):
    model_name = f'{config.data_config["dataset_name"]}-type_{args.mtype}-beta_{b}-vbr_0'
    loaded = torch.load(f"./model_params/{model_name}.pt", map_location=lambda storage, loc: storage)
    model.load_state_dict(loaded)
    print("model loaded")
    model = model.to(args.device)
    model.eval()
    for vi, v in enumerate(config.test_videos):
        data_folder = f'/home/ruihay1/playground/{v}'
        n_frames = len(os.listdir(data_folder))
        tot_psnr, tot_bpp, prev_frame = 0, 0, None
        for i in range(n_frames):
            img = Image.open(os.path.join(data_folder, f'{i+1}.png'))
            tensor = frame_processing(img)
            tensor = tensor.to(args.device)
            with torch.no_grad():
                if i % (config.gop_size if config.gop_size is not None else n_frames) == 0:
                    state = model.iframe_forward(tensor)
                else:
                    state = model.pframe_forward(tensor, prev_frame)
                prev_frame = state['output'].clamp(0, 1)
                tot_psnr += get_batch_psnr(state['output'].clamp(0, 1), tensor, 1)
                tot_bpp += state['bpp']
        print('video_type:', v, 'avg_psnr:', tot_psnr / n_frames, 'avg_bpp:', tot_bpp / n_frames)
        bpp_table[bi, vi] += tot_bpp.cpu().item() / n_frames
        psnr_table[bi, vi] += tot_psnr.cpu().item() / n_frames
if not os.path.isdir(config.test_output_path):
    os.mkdir(config.test_output_path)
np.save(os.path.join(config.test_output_path, f'{config.data_config["dataset_name"]}-type_{args.mtype}_bpp.npy'),
        bpp_table)
np.save(os.path.join(config.test_output_path, f'{config.data_config["dataset_name"]}-type_{args.mtype}_psnr.npy'),
        psnr_table)
