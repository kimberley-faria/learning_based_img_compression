import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from PIL import Image
from compressai.models import ScaleHyperprior
from compressai.zoo import bmshj2018_hyperprior
from torch import optim, nn, utils
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

device


class VAE_hyperprior(ScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "z_hat": z_hat,
            "scales_hat": scales_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


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
        print(x.shape)
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
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        layers = nn.ModuleList(list(backbone.children())[5:-1])

        self.in_channels = 32
        self.layer1_y_hat = self.make_layer(ResidualBlock, 128, 1)

        self.in_channels = 32
        self.layer1_scales_hat = self.make_layer(ResidualBlock, 128, 1)

        self.feature_extractor = nn.Sequential(*layers)

        # # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 23
        self.classifier = nn.Linear(2048 * 128, 23)

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
        print(y_hat.shape)
        y_hat = self.layer1_y_hat(y_hat)
        scales_hat = self.layer1_scales_hat(scales_hat)

        print(y_hat.shape, scales_hat.shape)

        x = torch.concat((y_hat, scales_hat))

        print(x.shape)
        x = torch.reshape(x, (128, 256, -1, 1))
        print(x.shape)

        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)

        print(representations.shape)

        x = self.classifier(representations.reshape((-1,)))

        print(x.shape)

        print(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y_hat, scales_hat, target = batch
        x, target = x.to(device), target.to(device)
        print(x.shape)

        # y_hat, scales_hat = y_hat.T, scales_hat.T
        y_hat, scales_hat = torch.squeeze(y_hat, 0), torch.squeeze(scales_hat, 0)

        print(y_hat.shape, y_hat.T.shape)

        y_hat, scales_hat = y_hat.to(device), scales_hat.to(device)

        predict = self.forward(y_hat.T, scales_hat.T)
        predict = torch.unsqueeze(predict, 0)
        print(predict.shape)
        print(target.shape)
        loss = F.cross_entropy(predict, target)
        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        print("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MINCDataset(data.Dataset):
    NUM_CLASS = 5

    def __init__(self, root=os.path.expanduser('.'),
                 train=True, transform=None, download=None):
        split = 'train' if train == True else 'val'
        root = os.path.join(root, 'minc-mini')
        print(root)
        self.transform = transform
        classes, class_to_idx = find_classes(root + '/images')
        if split == 'train':
            filename = os.path.join(root, 'labels/train1.txt')
        else:
            filename = os.path.join(root, 'labels/validate1.txt')

        self.images, self.y_hats, self.scales_hats, self.labels = make_dataset(filename, root,
                                                                               class_to_idx, transform=transform)
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _img = self.images[index]
        _label = self.labels[index]
        _y_hat = self.y_hats[index]
        _scales_hat = self.scales_hats[index]

        return _img, _y_hat, _scales_hat, _label

    def __len__(self):
        return len(self.images)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(filename, datadir, class_to_idx, transform=None):
    images = []
    y_hats = []
    scales_hats = []
    labels = []

    compression_model = VAE_hyperprior(128, 192)
    compression_model.load_state_dict(bmshj2018_hyperprior(quality=1, pretrained=True).state_dict())

    net = compression_model.eval().to(device)

    with open(os.path.join(filename), "r") as lines:
        for line in lines:
            _image = os.path.join(datadir, line.rstrip('\n'))
            _dirname = os.path.split(os.path.dirname(_image))[1]
            assert os.path.isfile(_image)
            label = class_to_idx[_dirname]

            _img = Image.open(_image).convert('RGB')
            if transform is not None:
                _img = transform(_img)
                _img = F.pad(input=_img, pad=(0, 150, 0, 150), mode='constant', value=0)

            print(_img.shape)
            images.append(_img)
            _img = _img.unsqueeze(0).to(device)

            with torch.no_grad():
                out_net = net.forward(_img)

            y_hat, scales_hat = out_net["y_hat"].to(device), out_net["scales_hat"].to(device)
            y_hats.append(y_hat)
            scales_hats.append(scales_hat)
            labels.append(label)

    return images, y_hats, scales_hats, labels


if __name__ == '__main__':
    downstream_model = cResnet39()
    downstream_model = downstream_model.to(device)

    minc_mini = MINCDataset(transform=ToTensor())
    train_loader = utils.data.DataLoader(minc_mini)

    trainer = pl.Trainer(limit_train_batches=1, max_epochs=1)
    trainer.fit(model=downstream_model, train_dataloaders=train_loader)
