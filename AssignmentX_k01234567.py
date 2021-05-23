#!/usr/bin/env python

from resources.config import Configuration, Grid, write_config

from resources.dataset import LayerSegmentation
import os
from torchvision import transforms
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1806)
torch.cuda.manual_seed(1806)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


@torch.no_grad()
def evaluate(network, data, loss, progress=None):
    network.eval()
    device = list(network.parameters())[0].device

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(device), targets.to(device)
            output = network(inputs)
            loss_batch = loss(output, targets)
            progress.track_loss_eval(loss_batch)

    return progress.summarise_loss_eval()


@torch.enable_grad()
def update(network, data, loss, opt, progress=None):
    device = list(network.parameters())[0].device
    for batch_idx, (inputs, targets) in enumerate(data):
        opt.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)

        output = network(inputs)
        loss_batch = loss(output, targets)

        loss_batch.backward()
        opt.step()

        progress.track_loss(loss_batch)

    return progress.summarise()


class Logger:
    """ Extracts and/or persists tracker information. """

    def __init__(self, path: str = None):
        """
        Parameters
        ----------
        path : str or Path, optional
            Path to where data will be logged.
        """
        path = Path() if path is None else Path(path)
        self.path = path.expanduser().resolve()

    def log_loss(self, epoch: int, update: int, loss: float):
        """
        Log the loss and other metrics of the current mini-batch.

        Parameters
        ----------
        epoch : int
            Rank of the current epoch.
        update : int
            Rank of the current update.
        loss : float
            Loss value of the current batch.
        """
        pass

    def log_summary(self, epoch: int, update: int, avg_loss: float):
        """
        Log the summary of metrics on the current epoch..

        Parameters
        ----------
        epoch : int
            Rank of the current epoch.
        update : int
            Rank of the current update.
        avg_loss : float
            Summary value of the current epoch.
        """
        pass

    def reset(self):
        if os.path.exists(f"{str(self.path)}/logs.txt"):
            os.remove(f"{self.path}/logs.txt")


class Tracker:
    """ Tracks useful information on the current epoch. """

    def __init__(self, *loggers: Logger, log_every: int = 1):
        """
        Parameters
        ----------
        logger0, logger1, ... loggerN : Logger
            One or more loggers for logging training information.
        log_every : int, optional
            Frequency of logging mini-batch results.
        """
        self.epoch = 0
        self.update = 0
        self.losses = []

        self.epoch_eval = 0
        self.update_eval = 0
        self.losses_eval = []

        self.loggers = list(loggers)
        self.log_every = log_every

    def track_loss(self, loss: float):
        # 0th epoch is to be used for evaluating baseline performance
        if self.epoch > 0:
            self.update += 1

        if self.log_every > 0 and self.update % self.log_every == 0:
            for logger in self.loggers:
                logger.log_loss(self.epoch, self.update, loss)

        self.losses.append(loss)

    def summarise(self):
        res = sum(self.losses) / max(len(self.losses), 1)
        self.losses.clear()

        for logger in self.loggers:
            logger.log_summary(self.epoch, self.update, res)

        self.epoch += 1
        return res

    def track_loss_eval(self, loss_eval):
        if self.epoch_eval > 0:
            self.update_eval += 1

        if self.log_every > 0 and self.update % self.log_every == 0:
            for logger in self.loggers:
                logger.log_loss_eval(self.epoch_eval, self.update_eval, loss_eval)

        self.losses_eval.append(loss_eval)

    def summarise_loss_eval(self):
        res = sum(self.losses_eval) / max(len(self.losses_eval), 1)
        self.losses_eval.clear()

        for logger in self.loggers:
            logger.log_summary_eval(self.epoch_eval, self.update_eval, res)

        self.epoch_eval += 1
        return res


class RelayNet(nn.Module):
    """ 
    Implementation of RelayNet model (segmentation).
    
    References
    ----------
    https://arxiv.org/abs/1704.02161
    """

    @staticmethod
    def build_block(in_channels: int, out_channels: int,
                    kernel_size: tuple = (7, 3), stride: int = 1):
        return nn.Sequential(
            # convolution with 'same' padding
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=tuple(k // 2 for k in kernel_size)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def __init__(self, in_channels: int = 1, num_classes: int = 5,
                 hid_channels: int = 64, scaling: int = 2):
        super().__init__()
        self.block1 = RelayNet.build_block(in_channels, hid_channels)
        self.pool = nn.MaxPool2d(scaling, return_indices=True)
        self.block2 = RelayNet.build_block(hid_channels, hid_channels)
        self.block3 = RelayNet.build_block(hid_channels, hid_channels)
        self.block4 = RelayNet.build_block(hid_channels, hid_channels)
        self.unpool = nn.MaxUnpool2d(scaling)
        self.block5 = RelayNet.build_block(2 * hid_channels, hid_channels)
        self.block6 = RelayNet.build_block(2 * hid_channels, hid_channels)
        self.block7 = RelayNet.build_block(2 * hid_channels, hid_channels)
        self.final_conv = nn.Conv2d(hid_channels, num_classes, 1)

    def forward(self, x):
        skip1 = self.block1(x)
        a1, idx1 = self.pool(skip1)
        skip2 = self.block2(a1)
        a2, idx2 = self.pool(skip2)
        skip3 = self.block3(a2)
        a3, idx3 = self.pool(skip3)
        a4 = self.block4(a3)
        up4 = self.unpool(a4, idx3, output_size=skip3.size())
        x5 = torch.cat([up4, skip3], dim=1)
        a5 = self.block5(x5)
        up5 = self.unpool(a5, idx2, output_size=skip2.size())
        x6 = torch.cat([up5, skip2], dim=1)
        a6 = self.block6(x6)
        up6 = self.unpool(a6, idx1, output_size=skip1.size())
        x7 = torch.cat([up6, skip1], dim=1)
        a7 = self.block7(x7)
        logits = self.final_conv(a7)
        return logits


class TensorboardAndProgressBarAndCheckpoint(Logger):
    " Log progress of epoch using a progress bar. "

    DESC_FORMAT = "Epoch {epoch: 3d}"
    METRIC_FORMAT = "{key:s}: {val:.5f}"
    SUMMARY_FORMAT = "(avg: {val:.3f})"

    DEFAULT_DIR = "runs"

    DEFAULT_NAME = "config{epoch:03d}"
    EXT = ".pth"

    def __init__(self, network, num_batches: int = None, path: str = None, tag: str = '', every: int = 1):
        super().__init__(self.DEFAULT_DIR if path is None else path)
        self._bar = tqdm(total=num_batches)
        self._bar.clear()

        self.writer = SummaryWriter(comment=tag)
        self.batch_idx = 0
        self.batch_idx_eval= 0

        self.network = network
        self.every = every

        if self.path.is_dir() or not self.path.suffix:
            self.path = self.path / self.DEFAULT_NAME
        self.path = self.path.with_suffix(self.EXT)
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self.tag = tag

    def log_loss(self, epoch: int, update: int, loss: float):
        self._bar.update()
        self._bar.set_description_str(self.DESC_FORMAT.format(epoch=epoch))
        self._bar.set_postfix_str(self.METRIC_FORMAT.format(key='loss', val=loss))

        self.writer.add_scalar('tag/loss', loss, self.batch_idx)
        self.batch_idx += 1

    def log_summary(self, epoch: int, update: int, avg_loss: float):
        self._bar.set_description_str(self.DESC_FORMAT.format(epoch=epoch))
        parts = [self._bar.postfix, self.SUMMARY_FORMAT.format(val=avg_loss)]
        self._bar.set_postfix_str(" ".join(parts))
        self._bar.display(msg="\n")

        prev_total = self._bar.n if self._bar.total is None else self._bar.total
        self._bar.reset(total=prev_total)
        self._bar.clear()

        self.writer.add_scalar('tag/avg_loss', avg_loss, epoch)

        filepath_weights = f"./checkpoints/{self.tag}{epoch}.pth"
        if epoch % self.every == 0:
            torch.save(self.network.state_dict(), filepath_weights)

    def log_loss_eval(self, epoch: int, update: int, loss_eval: float):
        self._bar.update()
        self._bar.set_description_str(self.DESC_FORMAT.format(epoch=epoch))
        self._bar.set_postfix_str(self.METRIC_FORMAT.format(key='test loss', val=loss_eval))

        self.writer.add_scalar('tag/loss_eval', loss_eval, self.batch_idx_eval)
        self.batch_idx_eval += 1

    def log_summary_eval(self, epoch: int, update: int, avg_loss_eval: float):
        self._bar.set_description_str(self.DESC_FORMAT.format(epoch=epoch))
        parts = [self._bar.postfix, self.SUMMARY_FORMAT.format(val=avg_loss_eval)]
        self._bar.set_postfix_str(" ".join(parts))
        self._bar.display(msg="\n")

        prev_total = self._bar.n if self._bar.total is None else self._bar.total
        self._bar.reset(total=prev_total)
        self._bar.clear()

        self.writer.add_scalar('tag/avg_loss_eval', avg_loss_eval, epoch)


os.system("rm -rf ./runs/")
os.system("mkdir -p checkpoints")

grid_epochs = Grid()
grid_epochs.add_options('lr', [1e-3, 1e-2, 1e-4])
grid_epochs.add_options('hs', [4,8])

grid = Grid(num_epochs=3, opt=grid_epochs)

# for conf in grid:
#     print(conf)


class segment:
    def __init__(self, hd, lr):
        self.model = RelayNet(hid_channels=hd).cuda()
        self.loss_func = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr=lr)


mean, std = 0.3299, 0.1746
normalise = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((mean,), (std,))
])

dataset_segmentation_train = LayerSegmentation("~/.pytorch", train=True, transforms=normalise)
loader_segmentation_train = DataLoader(dataset_segmentation_train, batch_size=20, shuffle=True, num_workers=10)
dataset_segmentation_test = LayerSegmentation("~/.pytorch", train=False, transforms=normalise)
loader_segmentation_test = DataLoader(dataset_segmentation_test, batch_size=8, shuffle=True, num_workers=10)
len_loader_train = len(loader_segmentation_train)
len_loader_test = len(loader_segmentation_test)
print(f"Train dataset length: {len_loader_train} \n Test dataset length {len_loader_test}")

for idx in range(len(grid)):
    try:
        config = grid[idx]
        print(config)

        inits = segment(config["opt"]["hs"], config["opt"]["lr"])

        tag = ""
        progress = TensorboardAndProgressBarAndCheckpoint(inits.model, len_loader_train, tag=str(config))
        tracker = Tracker(progress)
        for i in range(config["num_epochs"]):
            update(inits.model, loader_segmentation_train, inits.loss_func, inits.optimiser, progress=tracker)
            evaluate(inits.model, loader_segmentation_test, inits.loss_func, progress=tracker)

    except RuntimeError as err:
        import sys, traceback

        print(f"config {idx} failed to train", file=sys.stderr)
        traceback.print_exc()
