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
        if x.dim() == 3:
            old_dim1, old_dim2 = x.shape[0], x.shape[1]
            x = x.flatten(start_dim=0, end_dim=1)
        o = self.linear1(x)
        o = o.squeeze().reshape(old_dim1, old_dim2)
        return o
    
    def normalize(self):
        with torch.no_grad():
            w = self.linear1.weight
            w = F.normalize(w, dim=-1)
            self.linear1.weight.copy_(w)

class TwoLayerProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mid_dim = 1024
        self.linear1 = nn.Linear(d, self.mid_dim)
        self.linear2 = nn.Linear(self.mid_dim, 1)

    def forward(self, x):
        if x.dim() == 3:
            old_dim1, old_dim2 = x.shape[0], x.shape[1]
            x = x.flatten(start_dim=0, end_dim=1)
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        o = o.squeeze().reshape(old_dim1, old_dim2)
        return o


class ThreeLayerProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mid_dim = 1024
        self.linear1 = nn.Linear(d, self.mid_dim)
        self.linear2 = nn.Linear(self.mid_dim, self.mid_dim)
        self.linear3 = nn.Linear(self.mid_dim, 1)

    def forward(self, x):
        if x.dim() == 3:
            old_dim1, old_dim2 = x.shape[0], x.shape[1]
            x = x.flatten(start_dim=0, end_dim=1)
        h = F.relu(self.linear1(x))
        hh = F.relu(self.linear2(h))
        o = self.linear3(hh)
        o = o.squeeze().reshape(old_dim1, old_dim2)
        return o


class ProbeTrainer(object):
    def __init__(
        self,
        features,
        gt_labels,
        loss="ce",  # ce or pg
        weights=None,
        tr_pctg=1.0,
        seed=0,
        num_epochs=1000,
        ntries=1,
        lr=1e-4,
        batch_size=-1,
        verbose=False,
        device="cuda",
        layer=1,
        weight_decay=0.01,
    ):
        np.random.seed(seed)
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

        # data
        self.x_train = torch.tensor(
            x_train, dtype=torch.float, requires_grad=False, device=self.device
        )
        self.y_train = torch.tensor(
            y_train, dtype=torch.long, requires_grad=False, device=self.device
        )
        self.x_val = torch.tensor(
            x_val, dtype=torch.float, requires_grad=False, device=self.device
        )
        self.y_val = torch.tensor(
            y_val, dtype=torch.long, requires_grad=False, device=self.device
        )
        self.d = self.x_train.shape[-1]
        # print(self.x_train.shape, self.y_train.shape, self.x_val.shape, self.y_val.shape)
        # probe
        self.layer = layer
        self.initialize_probe()
        # self.best_probe = copy.deepcopy(self.probe)

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
        # p0, [B, 2]
        # p1, [B]
        # w, [B]
        if self.loss == "clip":
            tbs = (p1 * 2 - 1) * (p0[:, 1] - p0[:, 0])
            mse = torch.clamp(-tbs + .1, min=0)
            loss = torch.mean(w * mse)
        elif self.loss == "ce":
            tbs = (p1 * 2 - 1) * (p0[:, 1] - p0[:, 0])
            tbm = -F.logsigmoid(tbs)
            loss = torch.mean(w * tbm)
        return loss

    def get_acc(self, x, y):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        # x, y = self.x_val, self.y_val.tolist()
        batch_size = len(x) if self.batch_size == -1 else self.batch_size
        nbatches = len(x) // batch_size
        pred = []
        with torch.no_grad():
            for j in range(nbatches):
                x_batch = x[j * batch_size : (j + 1) * batch_size]
                y_batch = y[j * batch_size : (j + 1) * batch_size]

                # probe
                p_batch = self.probe(x_batch)  # [B, 2]
                y_val_hat = torch.argmax(p_batch, dim=-1).flatten().tolist()

                pred.extend(y_val_hat)
        hit = 0
        for y_hat, y in zip(pred, y):
            if y_hat == y:
                hit += 1
        acc = hit / (nbatches * batch_size)
        return acc

    def train(self):
        """
        Does a single training run of num_epochs epochs
        """
        acc_bucket = []
        loss_bucket = []
        permutation = torch.randperm(len(self.y_train))
        x, y = self.x_train[permutation], self.y_train[permutation]
        # x: [B, 2, F]
        # y: [B], 0 or 1, the index of which one wons

        if self.weights is not None:
            weights = self.weights[permutation]
        else:
            weights = torch.ones(len(y), device=self.device)

        # set up optimizer
        optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        batch_size = len(x) if self.batch_size == -1 else self.batch_size
        nbatches = len(x) // batch_size

        # Start training (full batch)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            for j in range(nbatches):
                x_batch = x[j * batch_size : (j + 1) * batch_size]
                y_batch = y[j * batch_size : (j + 1) * batch_size]
                w_batch = weights[j * batch_size : (j + 1) * batch_size]
                p_batch = self.probe(x_batch)  # [B, 2]
                loss = self.get_loss(p_batch, y_batch, w_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_bucket.append(loss.item())
            acc_bucket.append(self.get_acc(self.x_val, self.y_val.tolist()))
            if self.tr_pctg == 1.0:
                train_acc = acc_bucket[-1]
            else:
                train_acc = self.get_acc(self.x_train, self.y_train.tolist())
            pbar.set_description(
                f"Training probe, validation accuracy now = {acc_bucket[-1]*100:.2f}%, train accuracy = {train_acc*100:.2f}%"
            )

        return acc_bucket, loss_bucket
