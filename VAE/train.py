from model import *
from data import *
torch.backends.cuda.matmul.allow_tf32 = True

print('transform initializate sucsess')
train_dataset = Ego4d(img_dir='./data/',
                           transform1=transform1,
                           transform2=transform2)
print("train_dataset init")
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=14)
print("train_loader init")


lr = 0.001
epochs = 50
latent_dim = 32
model = VAE(latent_dim, batch_size=BATCH_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, epochs+1):
    x = next(iter(train_loader))
    model.train()
    print(f'Epoch {epoch} start')
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()
    
    model.eval()
    torch.save(model, f"VAE{epoch}.pth")