import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

class V(nn.Module):
    def __init__(self, nembed, nhidden=265, nlayers=1):
        super(V, self).__init__()

        self.nhidden = nhidden
        self.nlayers = nlayers
        
        self.lstm = nn.LSTM(nembed, nhidden, nlayers, batch_first=True)
        self.linear = nn.Linear(nhidden, nembed)
        
    def forward(self, x, h):
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        out = self.linear(out)
        return out, (h, c)
    
    def init_hidden(self, bsz):
        return (torch.zeros(self.nlayers, bsz, self.nhidden).to(DEVICE),
                torch.zeros(self.nlayers, bsz, self.nhidden).to(DEVICE))
    
def train_RNN(epochs, seqlen, model, z, ):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    # Train the model
    for epoch in range(epochs):
        # Set initial hidden and cell states
        hidden = model.init_hidden(32)
        loss = 0
        for i in range(0, 257, seqlen):
            # Get mini-batch inputs and targets
            inputs = z[:, i:i+seqlen, :].to(DEVICE)
            targets = z[:, (i+1):(i+1)+seqlen, :].to(DEVICE)
            
            # Forward pass
            hidden = detach(hidden)
            outputs, hidden = model(inputs, hidden)
            #print(outputs.shape, targets.shape)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
        if epoch % 50 == 0:
            print ('Epoch [{}/{}], Loss: {:.10f}'
                .format(epoch+1, epochs, loss))