import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        
        super().__init__()
        
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        
    def forward(self, x):
        
        #encode
        mean, log_var = self.encoder(x)
        
        #reparameterize
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        
        #decode
        x = self.decoder(z)
        
        return x, mean, log_var
    
class Encoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        
        super().__init__()
        
        self.enc_1 = nn.Linear(input_dim, 256)
        self.enc_21 = nn.Linear(256, latent_dim)
        self.enc_22 = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        
        x = F.relu(self.enc_1(x))
        mean = self.enc_21(x)
        log_var = self.enc_22(x)
        
        return mean, log_var
    
class Decoder(nn.Module):
    
    def __init__(self, latent_dim, output_dim):
        
        super().__init__()
        
        self.dec_1 = nn.Linear(latent_dim, 256)
        self.dec_2 = nn.Linear(256, output_dim)
        
    def forward(self, x):
    
        x = F.relu(self.dec_1(x))
        x = F.sigmoid(self.dec_2(x))    
            
        return x