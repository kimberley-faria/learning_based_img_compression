import sys

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from torch import utils
from torchvision.transforms import ToTensor
import wandb
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from PIL import Image
from compressai.models import ScaleHyperprior
from compressai.zoo import bmshj2018_hyperprior
from torch import optim, nn, utils
from torchvision import transforms
from torchmetrics import Accuracy

import wandb
from pytorch_lightning.loggers import WandbLogger
from compressai.losses import RateDistortionLoss
from compressai.models import ScaleHyperprior
from compressai.zoo import bmshj2018_hyperprior

import math

from dataset import MINCDataset2
from downstream_model import CombinedModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

if __name__ == '__main__':
    quality = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    inchannels = 320 if quality > 5 else 192
    num_epochs = 30
    batch_size = 32
    
    print("Quality: ", quality, ", Inchannels: ", inchannels, ", Epochs:", num_epochs, ", Batch size: ", batch_size)
    
    wandb_logger = WandbLogger(project='696ds-learning-based-image-compression', log_model=True)
    
    train_minc = MINCDataset2(train=True)
    val_minc = MINCDataset2(train=False)

    train_loader = utils.data.DataLoader(train_minc, batch_size=batch_size, shuffle=True)
    valdn_loader = utils.data.DataLoader(val_minc, batch_size=batch_size)

    downstream_model = CombinedModel(inchannels=inchannels, quality=quality)
    downstream_model = downstream_model.to(device)
    trainer = pl.Trainer(max_epochs=num_epochs, logger=wandb_logger)
    trainer.fit(model=downstream_model, train_dataloaders=train_loader, val_dataloaders=valdn_loader)
    
    wandb.finish()
