import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import optim, nn
from torchmetrics import Accuracy

import wandb
from pytorch_lightning.loggers import WandbLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, bias=False)


# Residual block
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, in_channels, stride)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = conv3x3(in_channels, in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = conv1x1(in_channels, out_channels, stride)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class cResnet39(pl.LightningModule):
    def __init__(self, inchannels=192):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        layers = nn.ModuleList(list(backbone.children())[5:-1])

        self.in_channels = inchannels
        self.layer1_y_hat = self.make_layer(ResidualBlock, 128, 1)

        self.in_channels = inchannels
        self.layer1_scales_hat = self.make_layer(ResidualBlock, 128, 1)

        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 23
        # self.classifier = nn.Linear(128*2048, 23)
        self.classifier = nn.Linear(2048, 23)
        
        
        
        self.training_targets = []
        self.validation_targets = []
        
        self.training_predictions = []
        self.validation_predictions = []
        
        self.training_step_losses = []
        self.validation_step_losses = []
        
        self.top1_accuracy = Accuracy(task="multiclass", num_classes=num_target_classes)
        self.top5_accuracy = Accuracy(task="multiclass", num_classes=num_target_classes, top_k=5)
        
        # save hyper-parameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()
        

    def make_layer(self, block, out_channels, blocks, stride=1):

        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = nn.ModuleList()
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        # print(*layers)
        return nn.Sequential(*layers)

    def forward(self, y_hat, scales_hat):
        y_hat = self.layer1_y_hat(y_hat.T)
        scales_hat = self.layer1_scales_hat(scales_hat.T)

        x = torch.concat((y_hat, scales_hat), 1)
        
        # self.feature_extractor.eval()
        # with torch.no_grad():
        representations = self.feature_extractor(x)

        representations = representations.view(representations.size(0), -1)
        
        output = self.classifier(representations)
        return output

    def training_step(self, batch, batch_idx):
        x, y_hat, scales_hat, target = batch
        target = target.to(device)

        y_hat, scales_hat = y_hat.to(device), scales_hat.to(device)
      
        predict = self.forward(y_hat.T, scales_hat.T)
        batch_loss = F.cross_entropy(predict, target)
        
        self.training_step_losses.append(batch_loss)

        self.training_targets.append(target)
        self.training_predictions.append(predict)
        
        return batch_loss
    
    def validation_step(self, batch, batch_idx):
        x, y_hat, scales_hat, target = batch
        
        target = target.to(device)

        y_hat, scales_hat = y_hat.to(device), scales_hat.to(device)
      
        with torch.no_grad():
            predict = self.forward(y_hat.T, scales_hat.T)
            batch_loss = F.cross_entropy(predict, target)
            
        self.validation_step_losses.append(batch_loss)

            
        self.validation_targets.append(target)
        self.validation_predictions.append(predict)
        
        return batch_loss
    
    def on_train_epoch_end(self):
        # print(torch.cat(self.training_targets).shape)
        # print(torch.cat(self.training_predictions).shape)
        
        loss = F.cross_entropy(torch.cat(self.training_predictions), torch.cat(self.training_targets))
        # loss1 = sum(self.training_step_losses) / len(self.training_step_losses)
        top1_accuracy = self.top1_accuracy(torch.cat(self.training_predictions), torch.cat(self.training_targets)) 
        top5_accuracy = self.top5_accuracy(torch.cat(self.training_predictions), torch.cat(self.training_targets)) 
        print("\nTrain loss:", loss)
        print("Train top-1 acc:", top1_accuracy)
        print("Train top-5 acc:", top5_accuracy)
        self.log("train loss", loss)
        self.log("train top-1 acc", top1_accuracy)
        self.log("train top-5 acc", top5_accuracy)
        self.training_targets.clear()
        self.training_predictions.clear()
        self.training_step_losses.clear()
        
    def on_validation_epoch_end(self):
        print(torch.cat(self.validation_targets).shape)
        print(torch.cat(self.validation_predictions).shape)
        
        loss = F.cross_entropy(torch.cat(self.validation_predictions), torch.cat(self.validation_targets))
        top1_accuracy = self.top1_accuracy(torch.cat(self.validation_predictions), torch.cat(self.validation_targets)) 
        top5_accuracy = self.top5_accuracy(torch.cat(self.validation_predictions), torch.cat(self.validation_targets)) 
        # loss = sum(self.validation_step_losses) / len(self.validation_step_losses)
        
        print("\nVal loss:", loss)
        print("Val top-1 acc:", top1_accuracy)
        print("Val top-5 acc:", top5_accuracy)
        self.log("val loss", loss)
        self.log("val top-1 acc", top1_accuracy)
        self.log("val top-5 acc", top5_accuracy)
        self.validation_targets.clear()
        self.validation_predictions.clear()
        self.validation_step_losses.clear()
    

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        
        # return optimizer
        
        
class Resnet50(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        layers = nn.ModuleList(list(backbone.children())[:-1])

        
        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 23
        # self.classifier = nn.Linear(128*2048, 23)
        self.classifier = nn.Linear(2048, 23)
        
        
        
        self.training_targets = []
        self.validation_targets = []
        
        self.training_predictions = []
        self.validation_predictions = []
        
        self.training_step_losses = []
        self.validation_step_losses = []
        
        self.top1_accuracy = Accuracy(task="multiclass", num_classes=num_target_classes)
        self.top5_accuracy = Accuracy(task="multiclass", num_classes=num_target_classes, top_k=5)
        
        # save hyper-parameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()
        

    def make_layer(self, block, out_channels, blocks, stride=1):

        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = nn.ModuleList()
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        # print(*layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        representations = self.feature_extractor(x)

        representations = representations.view(representations.size(0), -1)
        
        output = self.classifier(representations)
        return output

    def training_step(self, batch, batch_idx):
        x, x_hat, target = batch
        x_hat = x_hat.to(device)
        target = target.to(device)

      
        predict = self.forward(x_hat)
        batch_loss = F.cross_entropy(predict, target)
        
        self.training_step_losses.append(batch_loss)

        self.training_targets.append(target)
        self.training_predictions.append(predict)
        
        return batch_loss
    
    def validation_step(self, batch, batch_idx):
        x, x_hat, target = batch
        
        target = target.to(device)

        
      
        with torch.no_grad():
            predict = self.forward(x_hat)
            batch_loss = F.cross_entropy(predict, target)
            
        self.validation_step_losses.append(batch_loss)

            
        self.validation_targets.append(target)
        self.validation_predictions.append(predict)
        
        return batch_loss
    
    def on_train_epoch_end(self):
        # print(torch.cat(self.training_targets).shape)
        # print(torch.cat(self.training_predictions).shape)
        
        loss = F.cross_entropy(torch.cat(self.training_predictions), torch.cat(self.training_targets))
        # loss1 = sum(self.training_step_losses) / len(self.training_step_losses)
        top1_accuracy = self.top1_accuracy(torch.cat(self.training_predictions), torch.cat(self.training_targets)) 
        top5_accuracy = self.top5_accuracy(torch.cat(self.training_predictions), torch.cat(self.training_targets)) 
        print("\nTrain loss:", loss)
        print("Train top-1 acc:", top1_accuracy)
        print("Train top-5 acc:", top5_accuracy)
        self.log("train loss", loss)
        self.log("train top-1 acc", top1_accuracy)
        self.log("train top-5 acc", top5_accuracy)
        self.training_targets.clear()
        self.training_predictions.clear()
        self.training_step_losses.clear()
        
    def on_validation_epoch_end(self):
        print(torch.cat(self.validation_targets).shape)
        print(torch.cat(self.validation_predictions).shape)
        
        loss = F.cross_entropy(torch.cat(self.validation_predictions), torch.cat(self.validation_targets))
        top1_accuracy = self.top1_accuracy(torch.cat(self.validation_predictions), torch.cat(self.validation_targets)) 
        top5_accuracy = self.top5_accuracy(torch.cat(self.validation_predictions), torch.cat(self.validation_targets)) 
        # loss = sum(self.validation_step_losses) / len(self.validation_step_losses)
        
        print("\nVal loss:", loss)
        print("Val top-1 acc:", top1_accuracy)
        print("Val top-5 acc:", top5_accuracy)
        self.log("val loss", loss)
        self.log("val top-1 acc", top1_accuracy)
        self.log("val top-5 acc", top5_accuracy)
        self.validation_targets.clear()
        self.validation_predictions.clear()
        self.validation_step_losses.clear()
    

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        
        # return optimizer