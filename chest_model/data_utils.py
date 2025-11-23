'''
This script implements everything required to implement a dataset for images
'''

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 

from PIL import Image
import numpy as np



label_to_num = {
    'normal': 0, 
    'pneumonia': 1, 
    'tuberculosis': 2
}

class lung_dataset(Dataset):
    def __init__(self, 
                 im_files: list, 
                 transform: transforms = None):
        self.files = im_files
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, 
                   index: int):
        fp = self.files[index]
        label = fp.split('/')[-2]
        label = label_to_num[label]
        im = Image.open(fp)
        im = np.array(im)
        if self.transform != None:
            im = self.transform(im)
        return im, label


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.Resize((256, 256)),
    #transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])
