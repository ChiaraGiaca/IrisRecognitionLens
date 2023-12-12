from dataclasses import dataclass
from typing import Dict

import os
import torch
import torch.nn as nn
import random
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from iris_classifier_model import IrisClassifier, DATA_TRANSFORMS
from PIL import Image
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/content/drive/MyDrive/Biometric Systems/iris-recognition')
from utils import file_utils
#from pytorch_metric_learning import losses

# Directory with normalized photos splitted into train and val subsets
INPUT_DATA_DIR = '../data_iiitd/tmp/normalized_splitted'


@dataclass
class TrainConfig:
    data: Dict[str, ImageFolder] #dataset pieno (train, val)
    loaders: Dict[str, DataLoader] 
    criterion: nn.Module = nn.CrossEntropyLoss() 
    learning_rate: float = 0.0001
    num_epochs: int = 50
    

    def __post_init__(self):
        self.class_names = sorted(self.data['train'].classes)
        self.data_sizes = {x: len(self.data[x]) for x in ['train', 'val']}
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
        self.model = IrisClassifier(class_names=self.class_names,
                                    num_classes=len(self.class_names),
                                    load_from_checkpoint=False)
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.learning_rate
        )
        self.dataset = INPUT_DATA_DIR

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


#getting the images and dividing them in batches using the DataLoader
def create_train_config(batch_size: int = 16,
                        shuffle: bool = True,
                        num_workers: int = 4) -> TrainConfig:
    image_datasets = {
        x: ImageFolderWithPaths(os.path.join(INPUT_DATA_DIR, x), DATA_TRANSFORMS[x])
        for x in ['train', 'val']
    }
    
    data_loaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)
        for x in ['train', 'val']
    }


    return TrainConfig(image_datasets, data_loaders)
