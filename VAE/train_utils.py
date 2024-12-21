import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import torch
import tempfile
import torch.nn as nn
from torch import optim 
from torchinfo import summary
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable 

from tqdm import tqdm
import matplotlib.pyplot as plt

from VAE.model import VAE
from VAE.data import DEVICE

import ray
import os
import io


def load_dataloader(s3_link):
    dl = ray.data.read_binary_files(s3_link)
    buffer = io.BytesIO(dl.take(1)[0]['bytes'])
    dataloader = torch.load(buffer)

    print("loading dataloader compited )")

    return dataloader

def load_checkpoint(s3_link):
    cp = ray.data.read_binary_files(s3_link)
    buffer = io.BytesIO(cp.take(1)[0]['bytes'])
    checkpoint = torch.load(buffer)

    print("loading checkpoint compited )")
    return checkpoint

def train(config):
    s3checkpoint = config['s3checkpoint']
    s3dataloadder = config['s3dataloader']
    latent_dim = config['latent_dim']
    BATCH_SIZE = config['batchsize']
    epochs = config['epochs']
    lr = 0.001
    # loading checpoint of model
    checkpoint = load_checkpoint(s3checkpoint)
    # creating dataloader and preparate his to RAY
    train_loader = ray.train.torch.prepare_data_loader(load_dataloader(s3dataloadder))

    # init model
    model = VAE(latent_dim, batch_size=BATCH_SIZE).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = ray.train.torch.prepare_model(model)
    # set optim
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # set other virable
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    # train model

    for epoch in range(epoch, epochs+epoch):
        x = next(iter(train_loader))
        model.train()
        print(f'Epoch {epoch} start')
        eval_loss = 0
        # Loop through all batches in the training dataset
        for i, data, in enumerate(tqdm(train_loader)):
                data = data.to(DEVICE)
                optimizer.zero_grad()
                
                recon_batch, mu, logvar = model(data)
                loss = model.loss_function(recon_batch, data, mu, logvar)
                eval_loss += loss
                
                loss.backward() # Compute the gradients with respect to the model parameters
                
                optimizer.step() # Update the model parameters using the optimizer
        
        metrics = {"loss": loss.detach().cpu().numpy(), "epoch": epoch}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss':loss,
                                'epoch':epoch,
                                'full_model':model,
                                },
                                os.path.join(temp_checkpoint_dir, f"VAE_checkpoint_robot_{latent_dim}_{epoch+epochs}.pt")
            )
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
            
            
        if ray.train.get_context().get_world_rank() == 0:
            print(metrics)
