#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn, optim
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1806)
torch.cuda.manual_seed(1806)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class Tracker:
    """ Tracks useful information on the current epoch. """

    def __init__(self):
        self.losses = []

    def track_loss(self, loss: float):
        self.losses.append(loss)

    def summarise(self):
        avg = sum(self.losses) / len(self.losses)
        self.losses.clear()
        return avg





@torch.no_grad()
def evaluate(network, data, loss, progress: Tracker = None):
    network.eval()
    device = network._modules["0"].weight.device

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(device), targets.to(device)
            output = network(inputs)
            loss_batch = loss(output, targets)
            progress.track_loss(loss_batch)

    return progress.summarise()
    

@torch.enable_grad()
def update(network, data, loss, opt, progress: Tracker = None):
        
    for batch_idx, (inputs, targets) in enumerate(data):
        opt.zero_grad()
        
        output = network(inputs)
        loss_batch = loss(output, targets)
        
        loss_batch.backward()
        opt.step()

        progress.track_loss(loss_batch)

    return progress.summarise()


optimiser = optim.Adam(conv_net.parameters(), lr=1e-3)
loss_func = nn.CrossEntropyLoss()



# ## Logging Tracked Information
# 
# Thus far, the `Tracker` only keeps track of the loss throughout a single epoch.
# However, as mentioned earlier, a lot of features can be added to the tracking.
# For starters, it is often useful to keep track of how much epochs and/or update have already passed.
# 
# More importantly, we can use the `Tracker` to store certain information during training.
# Thus far, loss information has been collected to compute the average and is then discarded.
# In order to revisit this information later, it can be written to a file, or _logged_.
# 
# For this purpose, we will use the interface provided by the `Logger` class.
# This requires some modifications to the implementation of `Tracker`.
# However, these minor modifications will provide a powerful tool for monitoring.

# In[5]:


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


# In[6]:


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


# ### Exercise 2: Printing Progress (2 bonus bonus points)
# 
# Monitoring the loss early on during training can be useful
# to check whether things are working as expected.
# In combination with an indication of progress in training,
# expectations can be properly managed early on.
# 
#  > Create a logger that prints the progress in the current epoch (e.g., using percentages).
#  > It should also output the loss on the current mini-batch and what epoch is currently running.
#  > Finally, print the average loss after every epoch to keep track of performance over epochs.
# 
# **Hint:** In order to reduce the number of printed lines, you can make use of the carriage return, `\r`.
# In combination with setting `end=''` when calling the `print` function,
# this allows to move back to the beginning of the line and overwrite what was previously there.

# In[9]:


class Progress(Logger):
    " Log progress of epoch to stdout. "
    
    MSG_FORMAT = "\rEpoch {epoch:02d}: {it:3.0f}% - {key:s}: {val:.5f}"
    SUFFIX_FORMAT = "({avg_val:.5f})"

    def __init__(self, num_batches: int):
        """
        Parameters
        ----------
        num_batches : int
            The number of mini-batches in one epoch.
        """
        super().__init__()
        self.iter_count = num_batches
        self.counter = 0
        
    def log_loss(self, epoch: int, update: int, loss: float):
        with open(str(self.path) + "/logs.txt", "a") as file:
            file.write(f"\rEpoch {epoch}: {update}% - loss: {loss:.5f}")
            print(f"\rEpoch {epoch}: {update}% - loss: {loss:.5f}", end="")

    def log_summary(self, epoch: int, update: int, avg_loss: float):
        with open(str(self.path) + "/logs.txt", "a") as file:
            file.write(f"\rEpoch {epoch:02d}: {update:3.0f}% - average loss: {avg_loss:.5f}")
            print(f"\rEpoch {epoch:02d}: {update:3.0f}% - average loss: {avg_loss:.5f}", end="")


# In[10]:


# sanity check
progress = Progress(len(loader))
tracker = Tracker(progress)

evaluate(conv_net.cpu(), loader, loss_func, progress=tracker)
for _ in range(5):
    update(conv_net, loader, loss_func, optimiser, progress=tracker)


# ### Exercise 3: Tensorboard (2 points)
# 
# [Tensorboard](https://www.tensorflow.org/tensorboard) 
# is a library that allows to track and visualise data during and after training.
# Apart from scalar metrics, tensorboard can process distributions, images and much more.
# It started as a part of tensorflow, but was then made available as a standalone library.
# This makes it possible to use tensorboard for visualising pytorch data.
# As a matter of fact, tensorboard is readily available in pytorch.
# From [`torch.utils.tensorboard`](https://pytorch.org/docs/stable/tensorboard.html),
# the `SummaryWriter` class can be used to track various types of data.
# 
#  > Create a Logger that makes use of the `Summarywriter` to monitor the loss with tensorboard.
#  > On one hand, it should monitor the loss for every batch, e.g. using the tag `'tag/loss'`.
#  > On the other hand, it should monitor the average loss in every epoch, e.g. with the tag `'tag/avg_loss'`.

# In[17]:


class TensorBoard(Logger):
    """ Log loss values to tensorboard. """

    DEFAULT_DIR = "runs"
    
    def __init__(self, tag: str = '', path: str = None):
        """
        Parameters
        ----------
        tag : str, optional
            Tag for identifying the logged results.
        path : str or Path
            Path to the log directory.
        """
        super().__init__(TensorBoard.DEFAULT_DIR if path is None else path)
        self.writer = SummaryWriter()
        self.batch_idx = 0
    
    def log_loss(self, epoch: int, update: int, loss: float):
        self.writer.add_scalar('tag/loss', loss, self.batch_idx)
        self.batch_idx += 1
    
    def log_summary(self, epoch: int, update: int, res: float):
        self.writer.add_scalar('tag/avg_loss', res, epoch)


# In[19]:


# sanity check
tb = TensorBoard(tag="train")
tracker = Tracker(tb)
evaluate(conv_net, loader, loss_func, progress=tracker)
for _ in range(5):
    update(conv_net, loader, loss_func, optimiser, progress=tracker)


# In[23]:


get_ipython().run_line_magic('tensorboard', '--logdir runs')


# ### Exercise 4: Model Checkpoints (1 point)
# 
# Apart from logging metrics like e.g. loss and accuracy,
# it can often be useful to create checkpoints of the model.
# After all, you do not want hours of training to get lost
# due to a programming error in a print statement at the end of your code.
# If the full training would result in overfitting,
# this also allows to make use of the model before it started overfitting (cf. early stopping).
# 
#  > Implement a logger that saves the weights of the model every few epochs.
#  > Make sure to store all checkpoints in separate files!.
#  > Use the `.pth` extension for storing pytorch checkpoints.

# In[62]:


class Checkpoints(Logger):

    DEFAULT_NAME = "config{epoch:03d}"
    EXT = ".pth"
    
    def __init__(self, network: nn.Module, every: int = 10, path: str = None):
        super().__init__(path)
        self.network = network
        self.every = every

        if self.path.is_dir() or not self.path.suffix:
            # assume path is directory
            self.path = self.path / Checkpoints.DEFAULT_NAME
        # assure correct extension
        self.path = self.path.with_suffix(Checkpoints.EXT)
        # create directory if necessary
        self.path.parent.mkdir(exist_ok=True, parents=True)
    
    def log_summary(self, epoch: int, update: int, res: float):
        filepath_weights = str(self.path).split("{")[0]+str(epoch)+".pth"
        if epoch % self.every == 0:
            torch.save(self.network.state_dict(), filepath_weights)


# In[63]:


# sanity check
checkpoints = Checkpoints(conv_net, every=2, path="checkpoints")
tracker = Tracker(checkpoints)
for _ in range(5):
    update(conv_net, loader, loss_func, optimiser, progress=tracker)


# In[64]:


# sanity check
get_ipython().system(' ls checkpoints')


# In[65]:


# clean up checkpoints and tensorboard logs
get_ipython().system(' rm -r runs checkpoints')


# ## Hyperparameter Search
# 
# Finding good hyperparameters for a model is a general problem in machine learning (or even statistics).
# However, neural networks are (in)famous for their large number of hyperparameters.
# To list a few: learning rate, batch size, epochs, pre-processing, layer count, neurons for each layer, 
# activation function, initialisation, normalisation, layer type, skip connections, regularisation, ...
# Moreover, it is often not possible to theoretically justify a particular choice for a hyperparameter.
# E.g. there is no way to tell whether $N$ or $N + 1$ neurons in a layer would be better, without trying it out.
# Therefore, hyperparameter search for neural networks is an especially tricky problem to solve.

# ###### Manual Search
# 
# The most straightforward approach to finding good hyperparameters is to just 
# try out *reasonable* combinations of hyperparameters and pick the best model (using e.g. the validation set).
# The first problem with this approach is that it requires a gut feeling as to what *reasonable* combinations are.
# Moreover, it is often unclear how different hyperparameters interact with each other,
# which can make an irrelevant hyperparameter look more important than it actually is or vice versa.
# Finally, manual hyperparameter search is time consuming, since it is generally not possible to automate.

# ###### Grid Search
# 
# Getting a feeling for combinations of hyperparameters is often much harder than for individual hyperparameters.
# The idea of grid search is to get a set of *reasonable* values for each hyperparameter individually
# and organise these sets in a grid that represents all possible combinations of these values.
# Each combinations of hyperparameters in the grid can then be run simultaneously,
# assuming that so much hardware is available, which can speed up the search significantly.

# ###### Random Search
# 
# Since there are plenty of hyperparameters and each hyperparameters can have multiple *reasonable* values,
# it is often not feasible to try out every possible combination in the grid.
# On top of that, most of the models will be thrown away anyway because only the best model is of interest,
# even though they might achieve similar performance.
# The idea of random search is to randomly sample configurations, rather than choosing from pre-defined choices.
# This can be interpreted as setting up an infinite grid and trying only a few --- rather than all --- possibilities.
# Under the assumption that there are a lot of configurations with similarly good performance as the best model,
# this should provide a model that performs very good with high probability for a fraction of the compute.

# ###### Bayesian Optimisation 
# 
# Rather than picking configurations completely at random, 
# it is also possible to guide the random search.
# This is essentially the premise of Bayesian optimisation:
# sample inputs and evaluate the objective to find which parameters are likely to give good performance.
# 
# Bayesian optimisation uses a function approximator for the objective 
# and what is known as an *acquisition* function.
# The function approximator, or *surrogate*, 
# has to be able to model a distribution over function values, e.g. a Gaussian Process.
# The acquisition function then uses these distributions
# to find where the largest improvements can be made, e.g. using the cdf.
# For a more elaborate explanation of Bayesian optimisation, 
# see e.g. [this tutorial](https://arxiv.org/abs/1807.02811)
# 
# This approach is less parallellisable than grid or random search,
# since it uses the information from previous runs to find good sampling regions.
# However, often there are more configurations to be tried out than there are computing devices
# and it is still possible to sample multiple configurations at each step with Bayesian Optimisation.
# Also consider [this paper](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms) in this regard.

# ###### Neural Architecture Search
# 
# Instead of using Bayesian optimisation, 
# the problem of hyperparameter search can also be tackled by other optimisation algorithms.
# This approach is also known as *Neural Architecture Search* (NAS).
# There are different optimisation strategies that can be used for NAS,
# but the most common are evolutionary algorithms and (deep) reinforcement learning.
# Consider reading [this survey](http://jmlr.org/papers/v20/18-598.html) 
# to get an overview of how NAS can be used to construct neural networks.

# ## Real-World Deep Learning
# 
# In order to get a feeling for hyperparameter search,
# you have to try it out on some real world example.
# You can use the monitoring tools from previous exercises
# to log performance and get a feeling for which hyperparameters work well.
# 
# To get a progressbar for each epoch, you can use the logger below.
# Of course you are allowed to use your own code,
# but make sure to include all necessary code in this notebook.

# In[ ]:


class ProgressBar(Logger):
    " Log progress of epoch using a progress bar. "
    
    DESC_FORMAT = "Epoch {epoch: 3d}"
    METRIC_FORMAT = "{key:s}: {val:.5f}"
    SUMMARY_FORMAT = "(avg: {val:.3f})"

    def __init__(self, num_batches: int = None):
        super().__init__()
        self._bar = tqdm(total=num_batches)
        self._bar.clear()
        
    def log_loss(self, epoch: int, update: int, loss: float):
        self._bar.update()
        self._bar.set_description_str(self.DESC_FORMAT.format(epoch=epoch))
        self._bar.set_postfix_str(self.METRIC_FORMAT.format(key='loss', val=loss))
        
    def log_summary(self, epoch: int, update: int, avg_loss: float):
        self._bar.set_description_str(self.DESC_FORMAT.format(epoch=epoch))
        parts = [self._bar.postfix, self.SUMMARY_FORMAT.format(val=avg_loss)]
        self._bar.set_postfix_str(" ".join(parts))
        self._bar.display(msg="\n")
        
        prev_total = self._bar.n if self._bar.total is None else self._bar.total
        self._bar.reset(total=prev_total)
        self._bar.clear()


# ###### Configuration
# 
# In order to keep track of what combinations have been tried in hyperparmeter search,
# it is often useful to make use of configuration files.
# Another key advantage of keeping configuration files is reproducability.
# You can either write your own utilities (do not forget to include them here)
# or make use of the classes below.

# In[ ]:


# See documentation for examples on how to use these classes and functions
from resources.config import Configuration, Grid, write_config


# ###### Dataset
# 
# For the real-world exercise, we will use some medical imaging data.
# Each input image is a cross-section of the eye, with a focus on the retina.
# The goal is to automatically annotate a few layers within the retina.
# This task can be expressed as a segmentation task,
# where the regions between the layers are the classes to predict.
# 
# Each input sample is a 512x500 grayscale image.
# The mean intensity of the training images is 
# 0.3299 with standard deviation 0.1746.
# Since we have three annotated layers in the targets, 
# there are 4 different regions in the segmentation maps (labels 1-4).
# One additional class (label 0) has been used to mark "unlabelled" regions.
# 
# You can use the `LayerSegmentation` dataset for accessing the data.
# The constructor can be used to download and extract the ~3GB archives.
# By default, the dataset provides the images with segmentation maps as targets.
# It is also possible to get a more raw form of the targets,
# by setting `segmentation=False` in the constructor.
# The dataset should act like a regular torchvision dataset,
# i.e. the raw data is given through PIL images,
# and you can use `torchvision.transforms` to take care of pre-processing.

# In[ ]:


from resources.dataset import LayerSegmentation
raw_seg_data = LayerSegmentation("~/.pytorch", train=True) #, download=True)
img_seg = LayerSegmentation.to_image(*raw_seg_data[0])
raw_raw_data = LayerSegmentation("~/.pytorch", train=True, segmentation=False)
img_raw = LayerSegmentation.to_image(*raw_raw_data[0])


# In[ ]:


from PIL import Image
img = Image.new('RGB', (1000, 512))
img.paste(img_raw, (0, 0))
img.paste(img_seg, (500, 0))
display(img, metadata={'width': '100%'})


# ### Exercise 5: Student Gradient Descent (10 points)
# 
# Universities generally do not have the hardware to compete with industry.
# In order to get similar levels of parallellisation,
# universities make use of multiple students to try out hyperparameters 
# in a highly parallel fashion.
# 
#  > Find the best possible hyperparameters to train a network
#  > on the data included with this assignment.
#  > Use either grid search or random search 
#  > to test **at least five** different configurations.
#  > Make sure to try out different learning rates and architecture variations,
#  > since these are generally the most important hyperparameters.
#  > Store the parameters and configuration for the best performing model,
#  > since they will be part of your submission for the assignment.
#  > An implementation of the RelayNet network has been provided,
#  > but you are free to choose any network of your likings.
#  > Just make sure all code is in this notebook.
# 
# **Hint:** use a subset of the data and a low number of epochs to get a feeling for what choices work well initially. 

# In[ ]:


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

# TODO: add other utility functions here!


# In[ ]:


# TODO: set up different hyperparameters!
grid = Grid(num_epochs=5)
len(grid)


# In[ ]:


# when using tensorboard inline: use magic before training!
# %tensorboard --reload_multifile true --logdir


# In[ ]:


for idx in range(len(grid)):
    try:
        config = grid[idx]
        raise NotImplementedError("TODO: train network with hyper-parameters!")
    except RuntimeError as err:
        import sys, traceback
        print(f"config {idx} failed to train", file=sys.stderr)
        traceback.print_exc()

raise NotImplementedError("TODO: output best result + model (i.e. path to config/checkpoint)")

