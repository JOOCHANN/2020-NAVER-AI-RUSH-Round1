from os import listdir
from pathlib import Path
from tempfile import mkdtemp
from warnings import warn
import shutil
import pandas as pd
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import random
from PIL import Image
from torch.utils.data.dataset import Dataset

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
UNLABELED = -1
size = 430

#train_data
def train_transform(Rsize):
    return transforms.Compose([
        transforms.Resize((Rsize, Rsize), interpolation=Image.BICUBIC), #이미지 크기를 Rsize, Rsize로 Resizing함
        # transforms.RandomCrop(Rsize, padding=10),  # 이미지를 Crop한 뒤에 빈 곳을 padding함
        transforms.RandomAffine(degrees=0, scale=(0.80, 1.2)), # translate a = width(0.1), b = height(0)
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        # transforms.RandomGrayscale(p=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

#test_data
def test_transform(Rsize):
    return transforms.Compose([
        transforms.Resize((Rsize, Rsize), interpolation=Image.BICUBIC), #이미지 크기를 Rsize, Rsize로 Resizing함
        # transforms.RandomGrayscale(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

class SPAM(Dataset):
    def __init__(self, data_path, isTrain):
        super(SPAM, self).__init__()
        self.classes = ['normal', 'monotone', 'screenshot', 'unknown']
        self.base_dir = Path(mkdtemp())
        self.train = isTrain
        if self.train == True :
            print('Train data init')
            self._initialize_directory()
            self.output_dir = self.base_dir / 'train'
            self.src_dir = data_path / 'train'
            self.metadata = pd.read_csv(self.src_dir / 'train_label')
            self.image_filenames = []
            for _, row in self.metadata.iterrows():
                if row['annotation'] == UNLABELED:
                    continue
                src = self.src_dir / 'train_data' / row['filename']
                if not src.exists():
                    raise FileNotFoundError
                dst = self.output_dir / self.classes[row['annotation']] / row['filename']
                if dst.exists():
                    warn(f'File {src} already exists, this should not happen. Please notify 서동필 or 방지환.')
                else:
                    shutil.copy(src=src, dst=dst)
                
                self.image_filenames.append([dst, row['annotation']])

            self.transform = train_transform(Rsize = size)

        else :
            print('Test data init')
            # print('path', Path(data_path)) # /data/spam-1/test
            files = [str(p.name) for p in (Path(data_path) / 'test_data').glob('*.*') if p.suffix not in ['.gif', '.GIF']]
            self.metadata = pd.DataFrame({'filename': files})
            self.src_dir = Path(data_path) / 'test_data'
            print('test data path :', self.src_dir)
            self.image_filenames = []
            for _, row in self.metadata.iterrows():
                src = self.src_dir / row['filename']
                if not src.exists():
                    raise FileNotFoundError
                self.image_filenames.append([src, row['filename']])

            self.transform = test_transform(Rsize = size)
            print('test data :', len(self.image_filenames))

    def __del__(self):
        """
        Deletes the temporary folder that we created for the dataset.
        """
        shutil.rmtree(self.base_dir)

    def _initialize_directory(self) -> None:
        """
        Initialized directory structure for a given dataset, in a way so that it's compatible with the Keras dataloader.
        """
        dataset_path = self.base_dir / 'train'
        dataset_path.mkdir()
        for c in self.classes:
            (dataset_path / c).mkdir()


    def __getitem__(self, index):
        if self.train == True:
            image = self.transform(Image.open(self.image_filenames[index][0]))
            label = self.image_filenames[index][1]
            return image, label
        else :
            image = self.transform(Image.open(self.image_filenames[index][0]))
            files = self.image_filenames[index][1]
            return image, files

    def __len__(self):
        return len(self.image_filenames)

