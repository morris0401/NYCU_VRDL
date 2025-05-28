import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from net.model import PromptIR
from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor
import lightning.pytorch as pl
import torch.nn.functional as F
import torch.nn as nn
import os


def pad_input(input_, img_multiple_of=8):
    height, width = input_.shape[2], input_.shape[3]
    H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, \
        ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    return input_, height, width


def tile_eval(model, input_, tile=128, tile_overlap=32):
    b, c, h, w = input_.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h, w).type_as(input_)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
    restored = E.div_(W)
    restored = torch.clamp(restored, 0, 1)
    return restored


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss


# --- Helper functions for TTA ---
def augment_image(tensor_image, mode):
    if mode == 0:  # Original
        return tensor_image
    elif mode == 1:  # Horizontal flip
        return torch.flip(tensor_image, dims=[3])
    elif mode == 2:  # Vertical flip
        return torch.flip(tensor_image, dims=[2])
    elif mode == 3:  # Horizontal + Vertical flip
        return torch.flip(tensor_image, dims=[2, 3])
    else:
        raise NotImplementedError


def deaugment_image(tensor_image, mode):
    # Inverse of augment_image
    if mode == 0:  # Original
        return tensor_image
    elif mode == 1:  # Horizontal flip (is its own inverse)
        return torch.flip(tensor_image, dims=[3])
    elif mode == 2:  # Vertical flip (is its own inverse)
        return torch.flip(tensor_image, dims=[2])
    elif mode == 3:  # Horizontal + Vertical flip (is its own inverse)
        return torch.flip(tensor_image, dims=[2, 3])
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, \
                            1 for derain, 2 for dehaze, 3 for all-in-one')
    parser.add_argument('--test_path',
                        type=str,
                        default="test/demo/",
                        help='save path of test images, \
                            can be directory or an image')
    parser.add_argument('--output_path',
                        type=str,
                        default="output/demo/",
                        help='output save path')
    parser.add_argument('--ckpt_name',
                        type=str,
                        default="model.ckpt",
                        help='checkpoint save path')
    parser.add_argument('--tile',
                        type=bool,
                        default=False,
                        help="Set it to use tiling")
    parser.add_argument('--tile_size',
                        type=int,
                        default=128,
                        help='Tile size. None means \
                            testing on the original resolution image')
    parser.add_argument('--tile_overlap',
                        type=int,
                        default=32,
                        help='Overlapping of different tiles')
    # Add a TTA argument
    parser.add_argument('--tta',
                        action='store_true',
                        help='Enable Test Time Augmentation (flips)')
    opt = parser.parse_args()

    ckpt_path = "ckpt/" + opt.ckpt_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path, exist_ok=True)

    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.cuda)

    try:
        net = PromptIRModel.load_from_checkpoint(ckpt_path).to(device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure the checkpoint is compatible.")
        exit()
    net.eval()

    test_set = TestSpecificDataset(opt)
    testloader = DataLoader(test_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            num_workers=0)

    print('Start testing...')
    with torch.no_grad():
        for ([clean_name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device)

            if True:

                augmented_outputs = []

                tta_modes = [0, 1, 2, 3]

                for aug_mode in tta_modes:
                    # 1. Augment input
                    augmented_input = augment_image(degrad_patch,
                                                    aug_mode)

                    # Store original shape for unpadding if tiling
                    original_h, original_w = -1, -1

                    # 2. Process (with or without tiling)
                    if opt.tile is False:
                        restored_augmented = net(augmented_input)
                    else:
                        padded_augmented_input, h, w = \
                            pad_input(augmented_input)
                        original_h, original_w = h, w
                        restored_padded_augmented = tile_eval(
                            net,
                            padded_augmented_input,
                            tile=opt.tile_size,
                            tile_overlap=opt.tile_overlap
                        )
                        restored_augmented = \
                            restored_padded_augmented[:, :, :h, :w]

                    restored_deaugmented = deaugment_image(restored_augmented,
                                                           aug_mode)
                    augmented_outputs.append(restored_deaugmented)

                # 4. Average results
                restored = torch.stack(augmented_outputs).mean(dim=0)

            else:  # No TTA
                if opt.tile is False:
                    restored = net(degrad_patch)
                else:
                    degrad_patch_padded, h, w = pad_input(degrad_patch)
                    restored_padded = tile_eval(
                        net,
                        degrad_patch_padded,
                        tile=opt.tile_size,
                        tile_overlap=opt.tile_overlap
                    )
                    restored = restored_padded[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1)
            save_image_tensor(restored, os.path.join(opt.output_path,
                                                     clean_name[0] + '.png'))
    print('Testing finished.')
