import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy
from tqdm import tqdm


class LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 1)

    def forward(self, x):
        o = self.linear1(x)
        return o


class TwoLayerProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mid_dim = 256
        self.linear1 = nn.Linear(d, self.mid_dim)
        self.linear2 = nn.Linear(self.mid_dim, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return o


class ThreeLayerProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mid_dim = 256
        self.linear1 = nn.Linear(d, self.mid_dim)
        self.linear2 = nn.Linear(self.mid_dim, self.mid_dim)
        self.linear3 = nn.Linear(self.mid_dim, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        hh = F.relu(self.linear2(h))
        o = self.linear3(hh)
        return o


class ProbeTrainer(object):
    def __init__(
        self,
        features,
        gt_labels,
        loss="mse",  # mse or ce or pg
        weights=None,
        tr_pctg=1.0,
        seed=42,
        num_epochs=1000,
        ntries=1,
        lr=1e-4,
        batch_size=-1,
        verbose=False,
        device="cuda",
        layer=1,
        weight_decay=0.01,
        chunk_size=10,  # for pg
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if type(features) == list:
            gt_labels = np.array(gt_labels)
        total_sample = features.shape[0]
        train_idx = np.random.choice(
            range(len(features)), size=int(total_sample * tr_pctg), replace=False
        )

        self.tr_pctg = tr_pctg
        if tr_pctg == 1.0:
            val_idx = copy.deepcopy(train_idx)
        else:
            val_idx = np.array(list(set(range(len(features))) - set(train_idx)))
        x_train = features[train_idx]
        y_train = gt_labels[train_idx]
        x_val = features[val_idx]
        y_val = gt_labels[val_idx]

        # training
        self.device = device
        self.num_epochs = num_epochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.loss = loss
        self.chunk_size = chunk_size

        # data
        dt = torch.float
        if loss == "ce":
            self.loss_fn = torch.nn.BCELoss()

        self.x_train = torch.tensor(
            x_train, dtype=torch.float, requires_grad=False, device=self.device
        )
        self.y_train = torch.tensor(
            y_train, dtype=dt, requires_grad=False, device=self.device
        )
        self.x_val = torch.tensor(
            x_val, dtype=torch.float, requires_grad=False, device=self.device
        )
        self.y_val = torch.tensor(
            y_val, dtype=dt, requires_grad=False, device=self.device
        )
        self.d = self.x_train.shape[-1]
        # probe
        self.layer = layer
        self.initialize_probe()

        self.weights = weights.to(self.device) if weights is not None else None

    def initialize_probe(self):
        if self.layer == 1:
            self.probe = LinearProbe(self.d)
        elif self.layer == 2:
            self.probe = TwoLayerProbe(self.d)
        elif self.layer == 3:
            self.probe = ThreeLayerProbe(self.d)
        else:
            assert 0
        self.probe.to(self.device)

    def get_loss(self, p0, p1, w):
        if self.loss == "mse":
            mse = (p0.flatten() - p1.flatten()) ** 2
            loss = torch.mean(w * mse)
        elif self.loss == "ce":
            p0 = F.sigmoid(p0)
            loss = self.loss_fn(p0.flatten(), p1.flatten())
        elif self.loss == "pg":
            assert len(p0.shape) == 3
            probs = F.softmax(p0.squeeze(), dim=1)  # Using sample-dim for softmax
            rewards = p1
            assert probs.shape == rewards.shape
            loss = -(rewards * probs).sum(dim=1).mean()
        return loss

    def get_acc(self, x, y):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        batch_size = len(x) if self.batch_size == -1 else min(self.batch_size, len(x))
        nbatches = len(x) // batch_size
        pred = []
        with torch.no_grad():
            for j in range(nbatches):
                x_batch = x[j * batch_size : (j + 1) * batch_size]
                y_batch = y[j * batch_size : (j + 1) * batch_size]

                # probe
                p_batch = self.probe(x_batch)
                if self.loss == "mse":
                    y_val_hat = (1.0 * (p_batch > 0.5)).flatten().tolist()
                elif self.loss == "ce":
                    y_val_hat = (1.0 * (p_batch > 0)).flatten().tolist()
                elif self.loss == "pg":
                    y_val_hat = (1.0 * (p_batch > 0)).flatten().tolist()

                pred.extend(y_val_hat)
        hit = 0
        for y_hat, y in zip(pred, y):
            if y_hat == y:
                hit += 1
        acc = hit / (nbatches * batch_size)
        return acc

    def reshape_pg_data(self, x, y):
        # Shuffle and reshape data for PG loss
        x, y = self.x_train, self.y_train
        permutation = torch.randperm(x.shape[1])
        x, y = x[:, permutation], y[:, permutation]

        # Split into chunks
        xs = torch.split(x, self.chunk_size, dim=1)
        x = torch.concatenate(xs, dim=0)
        ys = torch.split(y, self.chunk_size, dim=1)
        y = torch.concatenate(ys, dim=0)

        permutation = torch.randperm(x.shape[0])
        x, y = x[permutation], y[permutation]
        return x, y

    def train(self):
        """
        Does a single training run of num_epochs epochs
        """
        acc_bucket = []
        loss_bucket = []
        permutation = torch.randperm(len(self.y_train))
        x, y = self.x_train[permutation], self.y_train[permutation]

        if self.loss == "pg":
            x, y = self.reshape_pg_data(x, y)

        if self.weights is not None:
            weights = self.weights[permutation]
        else:
            weights = torch.ones(len(y), device=self.device)

        # set up optimizer
        optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        batch_size = len(x) if self.batch_size == -1 else min(self.batch_size, len(x))
        nbatches = len(x) // batch_size

        # Start training (full batch)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            permutation = torch.randperm(len(self.y_train))
            x, y = self.x_train[permutation], self.y_train[permutation]
            if self.loss == "pg":
                x, y = self.reshape_pg_data(x, y)

            for j in range(nbatches):
                x_batch = x[j * batch_size : (j + 1) * batch_size]
                y_batch = y[j * batch_size : (j + 1) * batch_size]
                w_batch = weights[j * batch_size : (j + 1) * batch_size]
                p_batch = self.probe(x_batch)
                loss = self.get_loss(p_batch, y_batch, w_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_bucket.append(loss.item())
            if self.loss != "pg":
                acc_bucket.append(self.get_acc(self.x_val, self.y_val.tolist()))
                if self.tr_pctg == 1.0:
                    train_acc = acc_bucket[-1]
                else:
                    train_acc = self.get_acc(self.x_train, self.y_train.tolist())
                pbar.set_description(
                    f"Training probe, validation accuracy now = {acc_bucket[-1]*100:.2f}%, train accuracy = {train_acc*100:.2f}%"
                )
            else:
                p_val = self.probe(self.x_val)
                val_loss = self.get_loss(p_val, self.y_val, None)
                pbar.set_description(
                    f"Training probe, validation loss now = {val_loss:.4f}, train loss = {loss.item():.4f}"
                )

        return acc_bucket, loss_bucket
