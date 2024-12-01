import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

class MDNRNN(nn.Module):
    def __init__(self, z_size, n_hidden=256, n_gaussians=5, n_layers=1):
        super(MDNRNN, self).__init__()

        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(z_size, n_hidden, n_layers, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc2 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc3 = nn.Linear(n_hidden, n_gaussians*z_size)
        
    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.n_gaussians, self.z_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.z_size)
        
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma
        
        
    def forward(self, x, h):
        # Forward propagate LSTM
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)
    
    def init_hidden(self, bsz):
        return (torch.zeros(self.n_layers, bsz, self.n_hidden).to(DEVICE),
                torch.zeros(self.n_layers, bsz, self.n_hidden).to(DEVICE))

def mdn_loss_fn(y, pi, mu, sigma):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=2)
    loss = -torch.log(loss)
    return loss.mean()

def criterion(y, pi, mu, sigma):
    y = y.unsqueeze(2)
    return mdn_loss_fn(y, pi, mu, sigma)

def train_RNN(epochs, seqlen, model, z, bsz):
    optimizer = torch.optim.Adam(model.parameters())
    # Train the model
    for epoch in range(epochs):
        # Set initial hidden and cell states
        hidden = model.init_hidden(bsz)
        
        for i in range(0, 8, 16):
            # Get mini-batch inputs and targets
            inputs = z[:, i:i+seqlen, :].to(DEVICE)
            targets = z[:, (i+1):(i+1)+seqlen, :].to(DEVICE)
            
            # Forward pass
            hidden = detach(hidden)
            (pi, mu, sigma), hidden = model(inputs, hidden)
            loss = criterion(targets, pi, mu, sigma)
            
            # Backward and optimize
            model.zero_grad()
            loss.backward()
            # clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
        if epoch % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
                .format(epoch, epochs, loss.item()))