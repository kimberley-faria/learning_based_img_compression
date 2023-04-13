import sys

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from torch import utils
from torchvision.transforms import ToTensor
import wandb
from pytorch_lightning.loggers import WandbLogger

from dataset import MINCDataset
from downstream_model import cResnet39

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

if __name__ == '__main__':
    quality = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    inchannels = 320 if quality > 5 else 192
    num_epochs = 30
    batch_size = 32
    
    print("Quality: ", quality, ", Inchannels: ", inchannels, ", Epochs:", num_epochs, ", Batch size: ", batch_size)
    
    wandb_logger = WandbLogger(project='696ds-learning-based-image-compression', log_model=True)
    
    train_minc = MINCDataset(train=True, quality=quality)
    val_minc = MINCDataset(train=False, quality=quality)
    
    train_subset = list(range(0, len(train_minc), 3)) #16292 examples, 510 batches (32 bs)
    trainset = torch.utils.data.Subset(train_minc, train_subset)

    train_loader = utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valdn_loader = utils.data.DataLoader(val_minc, batch_size=batch_size)

    downstream_model = cResnet39(inchannels=inchannels)
    downstream_model = downstream_model.to(device)
    trainer = pl.Trainer(max_epochs=num_epochs, logger=wandb_logger)
    trainer.fit(model=downstream_model, train_dataloaders=train_loader, val_dataloaders=valdn_loader)
    
    wandb.finish()
