import torch
import numpy as np
from q_probe.trainer import ProbeTrainer


class ValueNetwork(object):
    def __init__(
        self,
        buffer_max_len=float("inf"),
        warmup_data=[],
        device="cuda:0",
        layer=1,
        lr=1e-4,
        num_epochs=1000,
        weigth_decay=0.01,
        tr_pctg=1.0,
        batch_size=-1,
        loss="mse",
        chunk_size=10,
        seed=42,
        **kwargs,
    ) -> None:
        self.buffer = []
        self.buffer_max_len = buffer_max_len
        self.device = device
        self.lr = lr
        self.layer = layer
        self.num_epochs = num_epochs
        self.weight_decay = weigth_decay
        self.probe = None
        self.tr_pctg = tr_pctg
        self.batch_size = batch_size
        self.loss = loss
        self.chunk_size = chunk_size
        self.seed = seed
        self.update_value(warmup_data)

    def get_value(self, act) -> float:
        # for [F], return a float number
        # for [N, F], return a N-long list
        assert self.probe is not None, "Probe not trained yet!"
        if type(act) is np.ndarray or type(act) is list:
            act = torch.tensor(act, dtype=torch.float)
        act = act.to(torch.float)
        act = act.to(self.device)
        if len(act.shape) == 1:
            act = act.unsqueeze(0)
        p = self.probe(act)
        p = p.detach().cpu().numpy()  # [N, 1]

        if len(p) == 1:
            p = p[0]
        return p

    def update_value(self, data, weights=None) -> None:
        # data: list of (hiddens, value)

        # handle buffer, and redo training
        if data is not None:
            self.buffer.extend(data)
            excess_length = len(self.buffer) - self.buffer_max_len
            if excess_length > 0:
                self.buffer = self.buffer[excess_length:]

        features = np.array([_[0] for _ in self.buffer])  # [N, F]
        if self.loss == "ce":
            gt_labels = np.array([_[1] for _ in self.buffer], dtype=np.int64)  # [N]
        else:
            gt_labels = np.array([_[1] for _ in self.buffer], dtype=np.float32)

        trainer = ProbeTrainer(
            features,
            gt_labels,
            weights=weights,
            num_epochs=self.num_epochs,
            device=self.device,
            lr=self.lr,
            layer=self.layer,
            weight_decay=self.weight_decay,
            tr_pctg=self.tr_pctg,
            batch_size=self.batch_size,
            loss=self.loss,
            chunk_size=self.chunk_size,
            seed=self.seed,
        )
        if self.probe is not None:  # initialize weights from previous training
            trainer.probe.load_state_dict(self.probe.state_dict())
        acc_bucket, loss_bucket = trainer.train()
        self.probe = trainer.probe
        del trainer
