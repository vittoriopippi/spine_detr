import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from .spine_transforms import *
from PIL import Image

class Spine2D(Dataset):

    def __init__(self, csv_file, img_folder, transform=None):
        self.all_vertebrae = pd.read_csv(csv_file)
        self.patients = self.all_vertebrae['patient_id'].drop_duplicates()
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.patients.iloc[idx]
        patient_vertebrae = self.all_vertebrae[self.all_vertebrae['patient_id'] == patient_id]

        img_path = self.img_folder / str(patient_id) / patient_vertebrae.iloc[0]['filename']

        import time
        t0 = time.time()

        # if os.path.exists('cache/' + img_path.name):
        #     img_path = 'cache/' + img_path.name
        #     image = io.imread(img_path)
        # else:
        #     image = io.imread(img_path)
        #     io.imsave('cache/' + img_path.name, image, check_contrast=False)
        image = io.imread(img_path)

        # print(f'Loaded in {round(time.time() - t0, 3)} s "{patient_vertebrae.iloc[0]["filename"]}"')
        image = np.interp(image, (image.min(), image.max()), (0, 255))
        image = np.stack((image,)*3, axis=-1)
        image = Image.fromarray(np.uint8(image))

        vertebrae = torch.tensor(patient_vertebrae[['vertebrae_id', 'center_x', 'center_y', 'genant_score']].values).float()

        info = {}
        info['spacing'] = patient_vertebrae.iloc[0]['spacing']
        info['patient_id'] = int(patient_id)
        info['src_width'], info['src_height'] = F._get_image_size(image)

        sample = {'image': image, 'vertebrae': vertebrae, 'info': info}

        if self.transform:
            sample = self.transform(sample)

        return sample

def build(image_set, args):
    root = Path(args.spine_ann_2d)
    assert root.exists(), f'provided 2D spine path {root} does not exist'

    if args.cross_val:
        PATHS = {
            "train": root / f"csv_train_{args.cross_val}.csv",
            "val": root / f"csv_test_{args.cross_val}.csv",
        }
    else:
        PATHS = {
            "train": root / f"csv_train.csv",
            "val": root / f"csv_test.csv",
        }
    
    TRANSFORMS = {
        "train": transforms.Compose([
                    RandomCrop(args.rand_crop) if args.rand_crop > 0 else FakeTransform(),
                    RandomRotation(args.rand_rot) if args.rand_rot > 0 else FakeTransform(),
                    RandomHorizontalFlip(args.rand_hflip) if args.rand_hflip > 0 else FakeTransform(),
                    # TODO add noise
                    Resize(args.resize) if args.resize > 0 else FakeTransform(),
                    ToTensor(),
                    ScaleCenters(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        "val": transforms.Compose([
                    RandomCrop(args.rand_crop) if args.rand_crop > 0 else FakeTransform(),
                    Resize(args.resize) if args.resize > 0 else FakeTransform(),
                    ToTensor(),
                    ScaleCenters(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
    }
    ann_file = PATHS[image_set]
    img_folder = Path(args.spine_imgs_2d)
    dataset = Spine2D(ann_file, img_folder, transform=TRANSFORMS[image_set])
    return dataset