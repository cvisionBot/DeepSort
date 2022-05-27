import os
import cv2
import glob
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class MarsDataset(Dataset):
    def __init__(self, path, transforms):
        super(MarsDataset, self).__init__()
        self.transforms = transforms
        self.data = glob.glob(path + '/mars-small128/bbox_train/**/*.jpg')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.imread(img_file)
        print(img_file)
        label = self.make_class_annotation(img_file)
        transformed = self.transforms(image=img)
        return {'img':transformed, 'class':label}
    
    def make_class_annotation(self, img_file):
        annot = img_file.split('/')
        annot = annot[-2]
        print(annot)
        return int(annot)


class Mars(pl.LightningDataModule):
    def __init__(self, path, workers, train_transforms, batch_size=None):
        self.path = path
        self.train_transforms = train_transforms
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader(MarsDataset(self.path, transforms=self.train_transforms),
                    batch_size=self.batch_size, num_workers=self.workers, persistent_workers=self.workers > 0, pin_memory=self.works)


if __name__ == '__main__':
    '''
    Dataset Loader Test
    run$ python -m dataset.deep.mars
    '''
    import albumentations
    import albumentations.pytorch
    # from dataset.deep.utils import visualize

    train_transforms = albumentations.Compose([
        albumentations.Resize(64, 128),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2()
    ])

    loader = DataLoader(MarsDataset(path='/mnt', transforms=train_transforms), batch_size=1, shuffle=True)

    for batch, sample in enumerate(loader):
        print(sample['img'])
        print(sample['class'])
        break
