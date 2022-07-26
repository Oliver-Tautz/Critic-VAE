from torch import nn
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from tqdm import trange, tqdm
from datasets import load_metric
import numpy as np
from crafter_extension_dataset import CrafterCriticDataset
from torch.utils.data import DataLoader


class Critic(nn.Module):

    def __init__(self, width=64, dims=[8, 8, 8, 16], bottleneck=32, colorchs=3, chfak=1, activation=nn.ReLU, pool="max",
                 dropout=0.5):
        """

        Taken from https://github.com/ndrwmlnk/critic-guided-segmentation-of-rewarding-objects-in-first-person-views/blob/main/nets.py
        """

        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.width = width
        stride = 1 if pool == "max" else 2
        dims = np.array(dims) * chfak
        pool = nn.MaxPool2d(2) if pool == "max" else nn.Identity()
        self.pool = pool
        features = [
            nn.Conv2d(colorchs, dims[0], 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(dims[0], dims[1], 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(dims[1], dims[2], 3, stride, 1),
            activation(),
            pool,
            nn.Dropout(dropout),
            nn.Conv2d(dims[2], dims[3], 3, stride, 1),
            activation(),
            pool,
            nn.Dropout(dropout),
            nn.Conv2d(dims[3], bottleneck * chfak, 4),
            activation()]
        self.features = nn.Sequential(*features)

        self.crit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chfak * bottleneck, chfak * bottleneck),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(chfak * bottleneck, 1)  # ,
            #            nn.Sigmoid()
        )

    def forward(self, X, collect=False):
        # takes shape of (batchsize,3, 64, 64)
        embeds = []
        # print(list(self.features))
        for layer in list(self.features):
            X = layer(X)
            if collect and isinstance(layer, type(self.pool)):
                embeds.append(X)
        if collect:
            embeds.append(X)
        # print("last embed", X.shape)
        pred = self.crit(X)

        if collect:
            return pred, embeds
        else:
            return pred

    def fit_on_crafter(self, X, Y, batch_size=32, epochs=2, dataset_size=50000, lossF=torch.nn.BCEWithLogitsLoss(),
                       optimizer=None, real=False, oversample=True):
        # X = torch.tensor of shape (N,3, 64, 64)
        # Y = torch.tensor of shape (N)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        if len(np.unique(Y)) == 2:
            train_x, eval_x, train_y, eval_y = train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=123456)
        else:
            train_x, eval_x, train_y, eval_y = train_test_split(X, Y, train_size=0.8, random_state=123456)

        if real:
            train_dataset = CrafterCriticDataset(train_x, train_y, oversample=oversample, dataset_size=dataset_size,
                                                 interpolate_real=True)
            eval_dataset = CrafterCriticDataset(eval_x, eval_y, interpolate_real=True)
        else:
            train_dataset = CrafterCriticDataset(train_x, train_y, oversample=oversample, dataset_size=dataset_size)
            eval_dataset = CrafterCriticDataset(eval_x, eval_y)

        del train_x, eval_x, train_y, eval_y

        if not optimizer:
            optimizer = AdamW(self.parameters())

        # if (device.type) != 'cpu':
        #   train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,pin_memory=True,pin_memory_device=device)
        #   eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size,pin_memory=True,pin_memory_device=device )

        # else:
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size)

        self.to(device)

        history = {'train_loss': [],
                   'val_loss': [],
                   'train_acc': [],
                   'val_acc': []}

        for e in trange(epochs, desc='train_epochs', position=0, leave=True):
            train_metric = load_metric("accuracy")
            self.train()
            batchloss = []
            for batch in tqdm(train_dataloader, desc='train_batches', position=0, leave=True):
                x, y = batch

                # print(x[0])
                # print(y[0:5])

                x = x.to(device)
                y = y.to(device).flatten()

                logits = self(x.float()).flatten()
                # print('\n\n-------',logits,'\n\n',self.sigmoid(logits),'-------\n\n')
                loss = lossF(logits, y.float())
                batchloss.append(loss.item())

                train_metric.add_batch(predictions=self.sigmoid(logits) >= 0.5, references=y)

                # print(f'loss={loss}')
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            history["train_acc"].append(train_metric.compute())
            history['train_loss'].append(np.mean(batchloss))

            self.eval()
            with torch.no_grad():
                eval_metric = load_metric("accuracy")
                batchloss_eval = []
                for batch in tqdm(eval_dataloader, desc='eval_batches', position=0, leave=True):
                    x, y = batch

                    x = x.to(device)
                    y = y.to(device).flatten()

                    logits = self(x.float()).flatten()
                    loss = lossF(logits, y.float())
                    batchloss_eval.append(loss.item())

                    eval_metric.add_batch(predictions=self.sigmoid(logits) >= 0.5, references=y)
                    # print(f'eval_loss={loss}')
            history["val_acc"].append(eval_metric.compute())
            history['val_loss'].append(np.mean(batchloss_eval))
        return history

    def evaluate(self, X, batchsize=None): # was called eval_intermediate
        
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.to(device)
        X = X.to(device)

        with torch.no_grad():
            # X = self.preprocess(X)

            if not batchsize:
                return self.forward(X, collect=False)
            else:
                out = []
                for batch in tqdm(DataLoader(X,batch_size=batchsize),desc='evaluate'):
                    batch.to(device)
                    out.append(self(batch))
                print(torch.vstack(out).shape)
                return torch.vstack(out)

