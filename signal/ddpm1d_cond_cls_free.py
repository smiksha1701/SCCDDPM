# Import necessary libraries and modules
import os
import copy
import sys
sys.path.insert(0, './modules/')
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules.modules1D_cond_cls_free import Unet1D_cond_cls_free, GaussianDiffusion1D_cond_cls_free
import logging
from torch.utils.tensorboard import SummaryWriter
from MITBIH import *
from torch.utils import data
import argparse

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# Define data root and load a pre-trained checkpoint
data_root = "./heartbeat/"
ckpt = torch.load("./checkpoint/DDPM1D_cls_free_MITBIH/checkpoint.pt")

# Define a function for training the model
def train(args):
    # Set up logging using the specified run name
    setup_logging(args.run_name)
    
    # Set the device to be used for training
    device = args.device
    
    # Load MITBIH dataset for ECG denoising
    ECG_denoising = mitbih_denosing(dataroot=data_root, class_id=0)
    dataloader = data.DataLoader(ECG_denoising, batch_size=32, num_workers=4, shuffle=True)
    
    # Initialize the model (Unet1D_cond_cls_free)
    model = Unet1D_cond_cls_free(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=args.num_classes,
        cond_drop_prob=0.5,
        self_condition=False,
        in_channels=2,
        out_channels=1
    ).to(device)
    
    # Skip the 'init_conv' layer weights from the loaded checkpoint
    temp_state_dict = ckpt['model_state_dict'].copy()
    temp_state_dict.pop("init_conv.weight", None)
    temp_state_dict.pop("init_conv.bias", None)

    # Load the modified state dictionary into the model
    #model.load_state_dict(temp_state_dict, strict=False)
    
    # Initialize GaussianDiffusion1D_cond_cls_free using the model
    diffusion = GaussianDiffusion1D_cond_cls_free(
        model,
        channels=1,
        seq_length=128,
        timesteps=1000,
        conditional=True
    ).to(device)
    
    # Initialize the AdamW optimizer for training the model
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Set up TensorBoard logging
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    
    # Get the total number of batches in the data loader
    l = len(dataloader)

    # Loop through epochs
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        # Use tqdm for progress bar visualization
        pbar = tqdm(dataloader)
        
        # Loop through batches in the data loader
        for i, data_dict in enumerate(pbar):
            # Move data to the specified device and set data types
            data_dict['ORG'] = data_dict['ORG'].to(device).to(torch.float)
            data_dict['COND'] = data_dict['COND'].to(device).to(torch.float)
            data_dict['Labels'] = data_dict['Labels'].to(device).to(torch.long)
            data_dict['Index'] = data_dict['Index'].to(device).to(torch.long)
            
            # Forward pass through the diffusion model and calculate loss
            loss = diffusion(data_dict, classes=data_dict['Labels'])

            # Zero out gradients, perform backward pass, and update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log the loss to TensorBoard
            logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)
        
        # Generate sampled signals for visualization
        class_id = 0
        labels = torch.tensor([0]*5).to(device)
        sampled_signals = diffusion.sample(
            x_in=data_dict['COND'][:5],
            classes=labels,
            cond_scale=3.
        )
        print(f'sampled_signals.shape {sampled_signals.shape}')  # (5, 1, 128)
        
        # Save sampled signals along with original and conditional signals
        is_best = False
        save_signals_cond_cls_free(
            sampled_signals,
            data_dict['ORG'][:5],
            data_dict['COND'][:5],
            labels,
            os.path.join("results", args.run_name, f"{epoch}.jpg")
        )
        
        # Save model checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join("checkpoint", args.run_name))

# Define a function to launch training based on command-line arguments
def launch(parser):
    args = parser.parse_args()
    args.device = "cuda:3"
    args.lr = 3e-4
    train(args)

# Entry point for script
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train DDPM1D_cond_cls_free model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes (default: 5)')
    parser.add_argument('--seq-length', type=int, default=128, help='Sequence Length (default: 128)')
    parser.add_argument('--run-name', type=str, default='DDPM1D_cond_cls_free', help='Run name to save (default: DDPM1D_cond_cls_free)')

    # Launch training
    launch(parser)
