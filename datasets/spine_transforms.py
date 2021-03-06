from torchvision import transforms
from torchvision.transforms import functional as F
import torch
import numbers
import numpy as np

class ToTensor():
    def __call__(self, sample):
        image, vertebrae = sample['image'], sample['vertebrae']
        return {'image': F.to_tensor(image), 'vertebrae': vertebrae, 'info': sample['info']}


class FakeTransform():
    """
    Placeholder transformation that doesn't do anything.
    """
    def __call__(self, sample):
        return sample


class RandomCrop(transforms.RandomCrop):
    def forward(self, sample):
        img, vertebrae = sample['image'], sample['vertebrae']

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        top, left, h, w = self.get_params(img, self.size)
        cropped_img = F.crop(img, top, left, h, w)

        vertebrae[:, 1] -= left 
        vertebrae[:, 2] -= top

        left_check = torch.logical_and(vertebrae[:, 1] < w, vertebrae[:, 1] >= 0) 
        top_check = torch.logical_and(vertebrae[:, 2] < h, vertebrae[:, 2] >= 0) 
        correct_vertebrae = torch.logical_and(top_check, left_check)
        vertebrae = vertebrae[correct_vertebrae]

        return {'image': cropped_img, 'vertebrae': vertebrae, 'info': sample['info']}


def center_crop(img, output_size):
    """
    This is the same function 'center_crop' inside functional
    which return also the parameters used for the crop
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_width, image_height = F._get_image_size(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        image_width, image_height = F._get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width), (crop_top, crop_left, crop_height, crop_width)

    
class CenterCrop(transforms.CenterCrop):
    def forward(self, sample):
        img, vertebrae = sample['image'], sample['vertebrae']
        cropped_img, coords = center_crop(img, self.size)

        top, left, h, w = coords

        vertebrae[:, 1] -= left 
        vertebrae[:, 2] -= top

        left_check = torch.logical_and(vertebrae[:, 1] < w, vertebrae[:, 1] >= 0) 
        top_check = torch.logical_and(vertebrae[:, 2] < h, vertebrae[:, 2] >= 0) 
        correct_vertebrae = torch.logical_and(top_check, left_check)
        vertebrae = vertebrae[correct_vertebrae]

        return {'image': cropped_img, 'vertebrae': vertebrae, 'info': sample['info']}


class FixedCrop:
    def __init__(self, coords, size):
        self.x, self.y = coords
        self.size = size

    def __call__(self, sample):
        img, vertebrae = sample['image'], sample['vertebrae']

        width, height = F._get_image_size(img)

        top, left, h, w = self.y, self.x, self.size, self.size
        cropped_img = F.crop(img, top, left, h, w)

        vertebrae[:, 1] -= left 
        vertebrae[:, 2] -= top

        left_check = torch.logical_and(vertebrae[:, 1] < w, vertebrae[:, 1] >= 0) 
        top_check = torch.logical_and(vertebrae[:, 2] < h, vertebrae[:, 2] >= 0) 
        correct_vertebrae = torch.logical_and(top_check, left_check)
        vertebrae = vertebrae[correct_vertebrae]

        return {'image': cropped_img, 'vertebrae': vertebrae, 'info': sample['info']}


class Resize(transforms.Resize):
    def forward(self, sample):
        assert isinstance(self.size, int), 'Tuple not supported yet'
        img, vertebrae = sample['image'], sample['vertebrae']
        width, height = F._get_image_size(img)
        
        if width < height:
            out_h = int(self.size * height / width)
            out_w = self.size
        else:
            out_w = int(self.size * width / height)
            out_h = self.size

        vertebrae[:, 1] = (vertebrae[:, 1] / width) * out_w
        vertebrae[:, 2] = (vertebrae[:, 2] / height) * out_h

        return {'image': F.resize(img, (out_h, out_w), self.interpolation), 'vertebrae': vertebrae, 'info': sample['info']}


class Normalize(transforms.Normalize):
    def forward(self, sample):
        img, vertebrae = sample['image'], sample['vertebrae']
        return {'image': F.normalize(img, self.mean, self.std, self.inplace), 'vertebrae': vertebrae, 'info': sample['info']}


class ScaleCenters():
    def __call__(self, sample):
        img, vertebrae = sample['image'], sample['vertebrae']
        c, h, w = img.shape
        vertebrae[:, 1] /= w
        vertebrae[:, 2] /= h
        return {'image': img, 'vertebrae': vertebrae, 'info': sample['info']}


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        img, vertebrae = sample['image'], sample['vertebrae']
        if torch.rand(1) < self.p:
            width, height = F._get_image_size(img)
            img = F.hflip(img)
            vertebrae[:, 1] = width - vertebrae[:, 1]
        return {'image': img, 'vertebrae': vertebrae, 'info': sample['info']}


class RandomRotation(transforms.RandomRotation):
    def __call__(self, sample):
        if self.degrees == 0:
            return sample

        img, vertebrae = sample['image'], sample['vertebrae']

        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        img = F.rotate(img, angle, self.resample, self.expand, self.center, fill)
        vertebrae[:, 1:3] = self.rotate_coord(img, angle, vertebrae[:, 1:3])

        width, height = F._get_image_size(img)

        x_check = torch.logical_or(vertebrae[:, 1] < 0, vertebrae[:, 1] >= width)
        y_check = torch.logical_or(vertebrae[:, 2] < 0, vertebrae[:, 2] >= height)
        xy_check = torch.logical_or(x_check, y_check)
        
        return {'image': img, 'vertebrae': vertebrae[torch.logical_not(xy_check)], 'info': sample['info']}

    @staticmethod
    def rotate_coord(img, angle, coords):
        x, y = coords.T
        width, height = F._get_image_size(img)
        cx, cy = width // 2, height // 2
        x -= cx
        y -= cy
        angle = (angle * np.pi) / 180
        
        r, theta = to_polar(x, y)
        theta -= angle
        x, y = to_cartesian(r, theta)

        x += cx
        y += cy
        return torch.stack([x, y]).T.int()

def to_polar(x, y):
    r = (x ** 2 + y ** 2).sqrt()
    theta = torch.atan(y / x)
    theta[x < 0] += np.pi
    return r, theta

def to_cartesian(r, theta):
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x, y