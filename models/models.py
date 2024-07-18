import torch
from torch import nn
import torch.nn.functional as F

### CVAE ###
class CEncoderSmall(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        super(CEncoderSmall, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(input_dim + n_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim), # 2 for mean and variance.
        )
        
        self.norm = torch.distributions.Normal(0, 1)
        self.norm.loc = self.norm.loc.cuda() # get sampling on the GPU
        self.norm.scale = self.norm.scale.cuda()
        
        self.kl = 0
        

    def forward(self, x, y):
        x = self.encode(torch.cat((x,y), dim=-1)) # concatenate label to input (conditioning), then encode
        mean, log_var = torch.chunk(x, 2, dim=-1)
        sigma = torch.exp(log_var)
        z = mean + sigma * self.norm.sample(mean.shape)
        self.kl = (sigma**2 + mean**2 - log_var - 0.5).sum() 
        return z
    
class CDecoderSmall(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        super(CDecoderSmall, self).__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), 
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        return self.decode(torch.cat((x,y), dim=-1))
    
class CVAESmall(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes, norm=True):
        super(CVAESmall, self).__init__()
        
        self.latent_dim = latent_dim
        self.norm = norm
        
        self.encoder = CEncoderSmall(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = CDecoderSmall(latent_dim, hidden_dim, input_dim, n_classes)
        self.n_classes = n_classes
        
    def forward(self, x, y):
        z = self.encoder(x, y)
        return self.decoder(z, y)

## larger CVAE ##
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        super(CEncoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim + n_classes, hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim, hidden_dim // 2)
        self.res_block3 = ResidualBlock(hidden_dim // 2, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 2 * latent_dim)  # 2 for mean and variance.
        
        self.norm = torch.distributions.Normal(0, 1)
        self.norm.loc = self.norm.loc.cuda()  # get sampling on the GPU
        self.norm.scale = self.norm.scale.cuda()
        
        self.kl = 0

    def forward(self, x, y):
        x = torch.cat((x, y), dim=-1)  # concatenate label to input (conditioning)
        x = F.relu(self.fc1(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.fc2(x)
        mean, log_var = torch.chunk(x, 2, dim=-1)
        sigma = torch.exp(log_var)
        z = mean + sigma * self.norm.sample(mean.shape)
        self.kl = (sigma**2 + mean**2 - log_var - 0.5).sum() 
        return z

class CDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        super(CDecoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim + n_classes, hidden_dim // 2)
        self.res_block1 = ResidualBlock(hidden_dim // 2, hidden_dim // 2)
        self.res_block2 = ResidualBlock(hidden_dim // 2, hidden_dim)
        self.res_block3 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, y):
        x = torch.cat((x, y), dim=-1)  # concatenate label to input (conditioning)
        x = F.relu(self.fc1(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = torch.sigmoid(self.fc2(x))
        return x
    
    
class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes, norm=True):
        super(CVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.norm = norm
        
        self.encoder = CEncoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = CDecoder(latent_dim, hidden_dim, input_dim, n_classes)
        self.n_classes = n_classes
        
    def forward(self, x, y):
        z = self.encoder(x, y)
        return self.decoder(z, y)
    
    
class CVAEWithKLAnnealing(CVAE):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes, norm=True, kl_start=0.01, kl_anneal_rate=0.01, kl_max=1.0):
        super(CVAEWithKLAnnealing, self).__init__(input_dim, hidden_dim, latent_dim, n_classes, norm)
        self.kl_start = kl_start
        self.kl_anneal_rate = kl_anneal_rate
        self.kl_weight = kl_start
        self.kl_max = kl_max

    def update_kl_weight(self, epoch):
        self.kl_weight = min(self.kl_max, self.kl_start + epoch * self.kl_anneal_rate)
        if self.kl_weight == self.kl_max:
            print(f'KL weight: max ({self.kl_max}) at epoch {epoch}')
        


### VAE ###
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim), # 2 for mean and variance.
        )
        
        self.norm = torch.distributions.Normal(0, 1)
        self.norm.loc = self.norm.loc.cuda() # get sampling on the GPU
        self.norm.scale = self.norm.scale.cuda()
        
        self.kl = 0
        

    def forward(self, x):
        x = self.encode(x) # concatenate label to input (conditioning), then encode
        mean, log_var = torch.chunk(x, 2, dim=-1)
        sigma = torch.exp(log_var)
        z = mean + sigma * self.norm.sample(mean.shape)
        self.kl = (sigma**2 + mean**2 - log_var - 0.5).sum() 
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decode(x)
    
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes, norm=True):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.norm = norm
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
        self.n_classes = n_classes
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



### Classifier ###
class N_Layer_Dense_Classifier(nn.Module):
    def __init__(self, num_feats, num_classes, num_layers, dropout_rate=0.0):
        super().__init__()
        layers = []
        for _ in range(num_layers -1):
            layers.extend([
                nn.Linear(num_feats, num_feats), 
                nn.ReLU(), 
                nn.Dropout(dropout_rate)
            ])
        layers.append(nn.Linear(num_feats, num_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    
    
    
    