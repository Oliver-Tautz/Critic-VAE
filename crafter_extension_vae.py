import torch
import torch.nn.functional as F
from torch import nn, Tensor
from vae_parameters import *
from math import exp
from vae_nets import MSSIM

class CrafterVariationalAutoencoder(nn.Module):
    def __init__(self, dims=[32, 64, 128, MAX_CHANNELS]):
        super(CrafterVariationalAutoencoder, self).__init__()
        self.encoder = CrafterVariationalEncoder(dims)
        self.decoder = CrafterDecoder(dims)
        self.mssim_loss = MSSIM()

    def forward(self, x, pred):
        mu, logvar = self.encoder(x)
        z_sample = self.reparametrize(mu, logvar)


        recon = self.decoder(z_sample, pred)




        """print('x: ', x.shape)
        print('mu: ', mu.shape)
        print('logvar: ', logvar.shape)
        print('recon: ', recon.shape)"""
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
        #print(x, mu, logvar, recon)


        torch.cuda.empty_cache()
        if CRAFTER_USE_MMSSIM:
            recon_loss = self.mssim_loss(recon, x)
        else:
            recon_loss = F.mse_loss(recon, x)

        recon_loss = torch.nan_to_num(recon_loss,0.99,0.99,-0.99)



        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        kld_loss *= kld_weight

        loss = recon_loss + kld_loss

        return {'total_loss': loss, 'recon_loss': recon_loss.detach(), 'KLD': kld_loss.detach()}


class CrafterVariationalEncoder(nn.Module):
    def __init__(self, dims):
        super(CrafterVariationalEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(ch, dims[0], k, step, p),  # to 64x64x32
            nn.BatchNorm2d(dims[0]),
            nn.MaxPool2d(2),  # to 32x32x32
            nn.ReLU(),

            nn.Conv2d(dims[0], dims[1], k, step, p),  # to 32x32x64
            nn.BatchNorm2d(dims[1]),
            nn.MaxPool2d(2),  # to 16x16x64
            nn.ReLU(),

            nn.Conv2d(dims[1], dims[2], k, step, p),  # to 16x16xint(MAX_CHANNELS/2)
            nn.BatchNorm2d(dims[2]),
            #nn.MaxPool2d(2),  # to 8x8xint(MAX_CHANNELS/2)
            nn.ReLU(),

            nn.Conv2d(dims[2], dims[3], k, step, p),  # to 8x8xMAX_CHANNELS
            nn.BatchNorm2d(dims[3]),
            #nn.MaxPool2d(2),  # to 4x4xMAX_CHANNELS
            nn.Tanh(),
        )

        # self.fcs = nn.Sequential(
        #    nn.Linear(bottleneck, bottleneck),
        #    nn.Tanh(),
        #    nn.Linear(bottleneck, bottleneck),
        #    nn.Tanh()
        # )

        # mu = mean, sigma = var; "fc" = fully connected layer
        self.fc_mu = nn.Linear(bottleneck, latent_dim)
        self.fc_var = nn.Linear(bottleneck, latent_dim)

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
            nn.Conv2d(dims[3], dims[2], k, step, p),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(dims[2], dims[1], k, step, p),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(dims[1], dims[0], k, step, p),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2),

            nn.Conv2d(dims[0], dims[0], k, step, p),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2),

            nn.Conv2d(dims[0], ch, k, step, p),
            nn.Tanh()  # tanh-range is [-1, 1], sigmoid is [0, 1]
        )

        self.decoder_input = nn.Linear(latent_dim + 1, bottleneck)
    def forward(self, z, pred, evalu=False, dim=1):

        if evalu:
            z = z[0]  # batch_size is 1 when evaluating
            dim = 0
        #print(self.decoder_input(torch.cat((z, pred), dim=dim)).shape)

        X = self.decoder_input(torch.cat((z, pred), dim=dim))
       #print(X.shape)

        X = X.view(-1,MAX_CHANNELS,int(h/4), int(w/4))

        X = self.model(X)

        return X




if __name__ == "__main__":
    encoder = CrafterVariationalEncoder([32, 64, int(MAX_CHANNELS/2), MAX_CHANNELS])
    decoder=CrafterDecoder([32, 64, int(MAX_CHANNELS/2), MAX_CHANNELS])
    X = torch.zeros((int(MAX_CHANNELS/2),3,64,64))
    pred = torch.zeros((int(MAX_CHANNELS/2),1))
    #print(pred.shape)
    def reparametrize(mu, logvar):  # logvar is variance
        std = torch.exp(0.5 * logvar)  # variance**2 = std
        eps = torch.randn_like(std)
        return mu + eps * std  # mean + random * standard-deviation

    mu, logvar = encoder(X)
    z_sample = reparametrize(mu, logvar)
    recon = decoder(z_sample, pred)
    #print(recon.shape)
