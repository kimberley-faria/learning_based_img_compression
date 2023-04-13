# We use the train-validation-test split
# 1 provided in the dataset, with 2125 training images, 125
# validation images and 250 testing images for each class.
import os
import sys
import pickle

import torch
import torch.nn.functional as F
from PIL import Image
from compressai.zoo import bmshj2018_hyperprior
from torch.utils import data
from torchvision import transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MINCDataset(data.Dataset):
    NUM_CLASS = 23

    def __init__(self, root=os.path.expanduser('../../'),
                 train=True, quality=1):
        split = 'train' if train == True else 'val'
        root = os.path.join(root, 'minc-2500')
        print(root)
        print("Quality:", quality)
        
        self.classes, self.class_to_idx = find_classes(root + '/images')
        if split == 'train':
            filename = os.path.join(root, 'labels/train1.txt') # 2125
        else:
            filename = os.path.join(root, 'labels/validate1.txt') # 125
            
        self.images, self.crs, self.labels = make_dataset(filename, root, self.class_to_idx, quality)
        
        self.compression_model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device)

        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _image = self.images[index]
        _img = Image.open(_image).convert('RGB')
        _label = self.labels[index]
        
        _img = transforms.ToTensor()(_img)
        _img = transforms.Resize(384)(_img)
    
        _img = _img.unsqueeze(0).to(device)
        
        _compressed_rep = self.crs[index]

        with open(_compressed_rep, 'rb') as handle:
            compressed_rep = pickle.load(handle)
        
        
        with torch.no_grad():
            _z_hat = self.compression_model.entropy_bottleneck.decompress(compressed_rep['strings'][1], compressed_rep['shape'])
            _scales_hat = self.compression_model.h_s(_z_hat)
            _indexes = self.compression_model.gaussian_conditional.build_indexes(_scales_hat)
            _y_hat = self.compression_model.gaussian_conditional.decompress(compressed_rep['strings'][0], _indexes, _z_hat.dtype)
        
        _y_hat, _scales_hat = torch.squeeze(_y_hat, 0).to(device), torch.squeeze(_scales_hat, 0).to(device)
        _y_hat = transforms.Resize(32)(_y_hat)
        _scales_hat = transforms.Resize(32)(_scales_hat)
        _y_hat = transforms.RandomCrop(28)(_y_hat)
        _scales_hat = transforms.RandomCrop(28)(_scales_hat)
        
        p = float(torch.randint(0, 2, (1, )).item())
        _y_hat = transforms.RandomHorizontalFlip(p=p)(_y_hat)
        _scales_hat = transforms.RandomHorizontalFlip(p=p)(_scales_hat)

        return _image, _y_hat, _scales_hat, _label

    def __len__(self):
        return len(self.images)


# We use the train-validation-test split
# 1 provided in the dataset, with 2125 training images, 125
# validation images and 250 testing images for each class.

class MINCDatasetDecoded(data.Dataset):
    NUM_CLASS = 23

    def __init__(self, root=os.path.expanduser('../../'),
                 train=True, quality=1):
        split = 'train' if train == True else 'val'
        root = os.path.join(root, 'minc-2500')
        print(root)
        
        self.classes, self.class_to_idx = find_classes(root + '/images')
        if split == 'train':
            filename = os.path.join(root, 'labels/train1.txt') # 2125
        else:
            filename = os.path.join(root, 'labels/validate1.txt') # 125

        self.images, self.crs, self.labels = make_dataset(filename, root, self.class_to_idx, quality, classic=True)
        
        self.compression_model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device)

        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _image = self.images[index]
        _img = Image.open(_image).convert('RGB')
        _label = self.labels[index]
        
        _img = transforms.ToTensor()(_img)
        _img = transforms.Resize(384)(_img)
    
        _img = _img.unsqueeze(0).to(device)
        
        _x_hat = self.crs[index]
        _x_hat = Image.open(_x_hat).convert('RGB')

        _x_hat = transforms.ToTensor()(_x_hat)
        _x_hat = transforms.Resize(256)(_x_hat)
        _x_hat = transforms.RandomCrop(224)(_x_hat)
        
        _x_hat = transforms.RandomHorizontalFlip()(_x_hat)
        

        return _image, _x_hat, _label

    def __len__(self):
        return len(self.images)


    


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(filename, datadir, class_to_idx, quality, classic=False):
    images = []
    labels = []
    crs = []
    
    i = 0
    with open(os.path.join(filename), "r") as lines:
        for line in lines:
            _image = os.path.join(datadir, line.rstrip('\n'))
            _dirname = os.path.split(os.path.dirname(_image))[1]
            _compressed_rep = _compressed_rep = os.path.join(datadir, 'enc_dec', f'bpp{quality}', _dirname, os.path.split(_image)[1]) if classic else os.path.join(datadir, 'compressed_rep', f'bpp{quality}', _dirname, os.path.splitext(os.path.split(_image)[1])[0])
            assert os.path.isfile(_image)
            assert os.path.isfile(_compressed_rep)
            label = class_to_idx[_dirname]
            images.append(_image)
            crs.append(_compressed_rep)
            labels.append(label)
            
            i += 1
            if i % 1000 == 0: sys.stdout.write('\r'+str(i)+' items loaded')
            
    sys.stdout.write('\r'+str(i)+' items loaded')
                           
              
    return images, crs, labels