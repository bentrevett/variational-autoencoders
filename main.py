import torch

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
N_EPOCHS = 10
INPUT_DIM = 28*28
LATENT_DIM = 2

transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST('./data', 
                               train=True, 
                               download=True, 
                               transform=transforms,
                               )

test_dataset = datasets.MNIST('./data',
                              train=False,
                              download=True,
                              transform=transforms,
                              )

train_iterator = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True
                            )

test_iterator = DataLoader(test_dataset,
                           batch_size=BATCH_SIZE,
                           )

def reconstruction_loss(x, reconstructed_x, mean, log_var):
    
    BCE = F.binary_cross_entropy(reconstructed_x, x, size_average=False)

    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    return BCE + KLD

model = models.VAE(INPUT_DIM, LATENT_DIM).to(device)

optimizer = optim.Adam(model.parameters())

def train():
    
    model.train()
    train_loss = 0
    
    for i, (x, _) in enumerate(train_iterator):
        x = x.view(-1, 28*28)
        x = x.to(device)
        optimizer.zero_grad()
        reconstructed_x, mean, log_var = model(x)
        loss = reconstruction_loss(x, reconstructed_x, mean, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    return train_loss

def test():
    
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        
        for i, (x, _) in enumerate(test_iterator):
            x = x.view(-1, 28*28)
            x = x.to(device)
            reconstructed_x, mean, log_var = model(x)
            loss = reconstruction_loss(x, reconstructed_x, mean, log_var)
            test_loss += loss.item()
        
    return test_loss

best_test_loss = float('inf')

for e in range(N_EPOCHS):
    
    train_loss = train()
    test_loss = test()
    
    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)
    
    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')