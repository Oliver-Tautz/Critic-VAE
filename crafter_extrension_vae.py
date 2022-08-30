import torch
import torch.nn.functional as F
from torch import nn, Tensor
from vae_parameters import *
from math import exp
from vae_nets import MSSIM

class CrafterVariationalAutoencoder(nn.Module):
    def __init__(self, dims=[32, 64, 128, 256]):
        super(CrafterVariationalAutoencoder, self).__init__()
        self.encoder = CrafterVariationalEncoder(dims)
        self.decoder = CrafterDecoder(dims)
        self.mssim_loss = MSSIM()

    def forward(self, x, pred):
        mu, logvar = self.encoder(x)
        z_sample = self.reparametrize(mu, logvar)
        recon = self.decoder(z_sample, pred)

        return x, mu, logvar, recon


    def recon_samples(self, x, reward):
        mu, logvar = self.encoder(x)
        recons = []
        for _ in range(6):
            sample = self.reparametrize(mu, logvar)
            recon = self.decoder(sample, reward)
            recons.append(recon)

        return recons

    def inject(self, x, reward=Tensor([0, 0.2, 0.4, 0.6, 0.8, 1])):
        reward = reward.to(device)
        mu, _ = self.encoder(x)

        recons = []
        for i in range(inject_n):
            recon = self.decoder(mu, reward[i].view(1), evalu=True)
            recons.append(recon)

        return recons

    def evaluate(self, x, pred):
        mu, _ = self.encoder(x)
        recon = self.decoder(mu, pred.view(1), evalu=True)

        return recon

    def reparametrize(self, mu, logvar):  # logvar is variance
        std = torch.exp(0.5 * logvar)  # variance**2 = std
        eps = torch.randn_like(std)
        return mu + eps * std  # mean + random * standard-deviation

    def vae_loss(self, x, mu, logvar, recon):

        torch.cuda.empty_cache()
        #recon_loss = F.mse_loss(recon, x)

        recon_loss = self.mssim_loss(recon, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        kld_loss *= kld_weight

        loss = recon_loss + kld_loss

        return {'total_loss': loss, 'recon_loss': recon_loss.detach(), 'KLD': kld_loss.detach()}


class CrafterVariationalEncoder(nn.Module):
    def __init__(self, dims):
        super(CrafterVariationalEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(ch, dims[0], k, step, p),  # to 64x64x32
            nn.ReLU(),
           # nn.BatchNorm2d(dims[0]),
            #nn.MaxPool2d(2),  # to 32x32x32
            nn.Conv2d(dims[0], dims[0], 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(dims[0], dims[1], k, step, p),  # to 32x32x64
            nn.ReLU(),
           # nn.BatchNorm2d(dims[1]),
           # nn.MaxPool2d(2),  # to 16x16x64
            nn.Conv2d(dims[1], dims[1], 3, 2, 1),
            nn.ReLU(),

            #nn.Conv2d(dims[1], dims[2], k, step, p),  # to 16x16x128
            #nn.ReLU(),
           # nn.BatchNorm2d(dims[2]),
            #nn.MaxPool2d(2),  # to 8x8x128
            #nn.Conv2d(dims[2], dims[2], 3, 2, 1),
            #nn.ReLU(),

            #nn.Conv2d(dims[2], dims[3], k, step, p),  # to 8x8x256
            #nn.ReLU(),
           # nn.BatchNorm2d(dims[3]),
            #nn.MaxPool2d(2),  # to 4x4x256
            #nn.Conv2d(dims[3], dims[3], 3, 2, 1),
            nn.Tanh(),
           # nn.ReLU(),
        )

        # self.fcs = nn.Sequential(
        #    nn.Linear(bottleneck, bottleneck),
        #    nn.Tanh(),
        #    nn.Linear(bottleneck, bottleneck),
        #    nn.Tanh()
        # )

        # mu = mean, sigma = var; "fc" = fully connected layer
        self.fc_mu = nn.Linear(CRAFTER_BOTTLENECK, latent_dim)
        self.fc_var = nn.Linear(CRAFTER_BOTTLENECK, latent_dim)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)


        z_flat = torch.flatten(x, start_dim=1)
        # z_flat = self.fcs(z_flat)

        mu = self.fc_mu(z_flat)
        log_var = self.fc_var(z_flat)

        return mu, log_var


class CrafterDecoder(nn.Module):
    def __init__(self, dims):
        super(CrafterDecoder, self).__init__()
        self.model = nn.Sequential(
            #nn.Conv2d(dims[3], dims[2], k, step, p),
            #torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=3, padding=2, dilation=1),
            #nn.ReLU(),
            #nn.Upsample(scale_factor=2),


            #nn.Conv2d(dims[2], dims[1], k, step, p),
            #torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=3, padding=4, dilation=1),
            #nn.ReLU(),
            #nn.Upsample(scale_factor=2),

            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=3, padding=8, dilation=1),
            #nn.Conv2d(dims[1], dims[0], k, step, p),,
            nn.ReLU(),
            #nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=3, padding=16, dilation=1),
            #nn.Conv2d(dims[0], dims[0], k, step, p),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2),

            nn.Conv2d(dims[0], ch, k, step, p),
            nn.Tanh()  # tanh-range is [-1, 1], sigmoid is [0, 1]
            #nn.Sigmoid()
        )

        self.decoder_input = nn.Linear(latent_dim + 1, CRAFTER_BOTTLENECK)

    def forward(self, z, pred, evalu=False, dim=1):

        if evalu:
            z = z[0]  # batch_size is 1 when evaluating
            dim = 0
        #print(self.decoder_input(torch.cat((z, pred), dim=dim)).shape)

        X = self.decoder_input(torch.cat((z, pred), dim=dim))
        print(X.shape)

        X = X.view(-1, MAX_CHANNELS, BOTTLENECK_DIM, BOTTLENECK_DIM)

        X = self.model(X)

        return X




if __name__ == "__main__":
    encoder = CrafterVariationalEncoder([32, 64, 128, 256])
    decoder=CrafterDecoder([32, 64, 128, 256])
    X = torch.zeros((128,3,64,64))
    pred = torch.zeros((128,1))
    print(pred.shape)
    def reparametrize(mu, logvar):  # logvar is variance
        std = torch.exp(0.5 * logvar)  # variance**2 = std
        eps = torch.randn_like(std)
        return mu + eps * std  # mean + random * standard-deviation

    mu, logvar = encoder(X)
    z_sample = reparametrize(mu, logvar)
    recon = decoder(z_sample, pred)
    #print(recon.shape)
