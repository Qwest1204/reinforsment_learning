import torch
import torch.nn as nn
from torch import optim 
from torchinfo import summary
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable 

from tqdm import tqdm
import matplotlib.pyplot as plt

from VAE.model import VAE
from VAE.data import Ego4d, DEVICE, BATCH_SIZE, transform1, transform2, ResumableRandomSampler

checkpoint = torch.load('/home/qwest/project/PycharmProjects/Reinforsment_Learning/VAE/weights/main/VAE_checkpoint_32_69.pt')

print('transform initializate sucsess')
train_dataset = Ego4d(img_dir='/home/qwest/data_for_ml/3_25',
                           transform1=transform1,
                           transform2=transform2)
print("train_dataset init")
sampler = ResumableRandomSampler(train_dataset)
sampler.set_state(checkpoint['sampler_state'])
print("train_sampler init")
train_loader = DataLoader(dataset=train_dataset,
                           batch_size=32,
                           shuffle=False,
                           sampler=sampler,
                           num_workers=6)
print("train_loader init")

print("Len of trainloader: ",len(train_loader))

lr = 0.001
epochs = 70
latent_dim = 32

model = VAE(latent_dim, batch_size=32).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=lr)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print("total ep: ", epoch)
print("total loss: ", loss)
print("optimizer: ", optimizer.state)

def train(epoch):
        """
        train VAE model.

        Args:
        epoch (int): number of epoch.
        """

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

        torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss':loss,
                        'epoch':epoch,
                        'full_model':model,
                        'sampler_state':sampler.get_state(),
                        },
                        f'VAE/weights/main/VAE_checkpoint_{latent_dim}_{epoch}.pt')
        
        print(f"Avg loss: {loss:2f} \n")

# Log model summary.|
with open("model_summary.txt", "w") as f:
    f.write(str(summary(model)))

for t in range(epoch+1, epochs+1):
    train(t)
