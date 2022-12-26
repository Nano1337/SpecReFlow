import torchsummary 
import torch
import os
from models.unet import UNet
import contextlib

def make_torch_summary(output_dir):
    """
    Get Unet architecture information for figure 3
    """
    model = UNet(in_channels=3,
            out_channels=1,
            n_blocks=4,
            start_filters=8,
            activation='relu',
            normalization='batch',
            conv_mode='same',
            dim=2
            ).to('cuda')
    
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        with contextlib.redirect_stdout(f):
            torchsummary.summary(model, (3, 288, 384))

def make_fig_3():
    output_dir = "../../figs/fig_3/"
    make_torch_summary(output_dir)

if __name__ == "__main__":
    make_fig_3()