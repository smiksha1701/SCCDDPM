import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules1D_cond_cls_free import Unet1D_cond_cls_free, GaussianDiffusion1D_cond_cls_free
import logging
from torch.utils.tensorboard import SummaryWriter
from MITBIH import *
from torch.utils import data
import argparse

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
data_root = "../heartbeat/"
ckpt = torch.load("./checkpoint/DDPM1D_cls_free_MITBIH/checkpoint.pt")


def train(args):
    setup_logging(args.run_name)
    device = args.device
    ECG_denoising = mitbih_denosing(dataroot = data_root, class_id=0)
    dataloader = data.DataLoader(ECG_denoising, batch_size=32, num_workers=4, shuffle=True)
    
    model = Unet1D_cond_cls_free(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = args.num_classes,
        cond_drop_prob = 0.5,
        self_condition = False,
        in_channels = 2,
        out_channels = 1).to(device)
    
    # skip the 'init_conv' layer weights
    # Create a temporary copy of the state dictionary
    temp_state_dict = ckpt['model_state_dict'].copy()
    # print(temp_state_dict.keys())

    # Remove the weights for the 'init_conv' layer
    temp_state_dict.pop("init_conv.weight", None)
    temp_state_dict.pop("init_conv.bias", None)


    #model.load_state_dict(temp_state_dict, strict=False)
    diffusion = GaussianDiffusion1D_cond_cls_free(
        model,
        channels = 1,
        seq_length = 128,
        timesteps = 1000,
        conditional = True).to(device)
    
    
    
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, data_dict in enumerate(pbar):
            
            data_dict['ORG'] = data_dict['ORG'].to(device).to(torch.float)
            data_dict['COND'] = data_dict['COND'].to(device).to(torch.float)
            data_dict['Labels'] = data_dict['Labels'].to(device).to(torch.long)
            data_dict['Index'] = data_dict['Index'].to(device).to(torch.long)
            
            loss = diffusion(data_dict, classes = data_dict['Labels'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)
            
        class_id=0
        labels = torch.tensor([0]*5).to(device)

        sampled_signals = diffusion.sample(
            x_in = data_dict['COND'][:5],
            classes = labels,
            cond_scale = 3.)
        print(f'sampled_signals.shape {sampled_signals.shape}') # (5, 1, 128)
        
        is_best = False
        
        save_signals_cond_cls_free(sampled_signals, data_dict['ORG'][:5], data_dict['COND'][:5], labels, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join("checkpoint", args.run_name))


def launch(parser):
    
    args = parser.parse_args()
    args.device = "cuda:3"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DDPM1D_cond_cls_free model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes (default: 5)')
    parser.add_argument('--seq-length', type=int, default=128, help='Seqence Length (default: 128)')
    parser.add_argument('--run-name', type=str, default='DDPM1D_cond_cls_free', help='Run name to save (default: DDPM1D_cond_cls_free)')
    

    launch(parser)


