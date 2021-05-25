from models import build_model
from main import get_args_parser
from PIL import Image
from skimage import io
from datasets.spine_transforms import *
from torchvision import transforms
import torch
from torchvision.transforms import functional as F
import argparse
from util.spine_plot import *

class SpineDETR:
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.model, _, _ = build_model(args)
        self.model.to(self.device)

        checkpoint = torch.load(args.resume, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])

        self.model.eval()
    
    def __call__(self, samples):
        samples = samples.to(self.device)
        return self.model(samples)

def batch_gen(img_path, stride=None):
    image = io.imread(img_path)
    image = np.interp(image, (image.min(), image.max()), (0, 255))
    image = np.stack((image,)*3, axis=-1)
    image = Image.fromarray(np.uint8(image))

    window_size = args.rand_crop
    stride = window_size // 2 if stride is None else stride
    width, height = F._get_image_size(image)

    w_list = [x for x in range(0, width, stride) if (x + window_size) <= width]
    h_list = [y for y in range(0, height, stride) if (y + window_size) <= height]

    w_list.append(width - window_size)
    h_list.append(height - window_size)

    wh_list = [(w, h) for w in w_list for h in h_list]

    images = []
    for x, y in wh_list:
        transform = transforms.Compose([
                        FixedCrop((x, y), window_size),
                        Resize(224),
                        ToTensor(),
                        ScaleCenters(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

        sample = {'image': image, 'vertebrae': torch.zeros([1, 4]).float(), 'info': {}}
        sample = transform(sample)
        images.append(sample['image'])

    batch = torch.stack(images)
    return batch, wh_list, image

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model = SpineDETR(args)

    batch, windows, src_img = batch_gen('//archive.ostfalia.de/I-ARTEMIS/Data/Diagnostikbilanz/Fertig 20190503/1001/1001_UKSH_KIEL_RADIOLOGIE_NEURORAD_KVP80_cExp129.399_PixSp0-1_20Transversals.tif')

    out = model(batch)

    window_size = args.rand_crop
    w_centers = torch.Tensor(windows).to(args.device)
    w_centers += window_size // 2

    out_centers = []
    for i, (window, logits, centers) in enumerate(zip(windows, out['pred_logits'], out['pred_boxes'])):
        centers = centers[logits.squeeze() > 0.5]
        centers = centers * window_size
        centers[:, 0] += window[0]
        centers[:, 1] += window[1]
        if centers.numel() > 0:
            dist = torch.cdist(centers, w_centers)
            min_idx = dist.argmin(1)
            correct_centers = centers[min_idx == i]
            out_centers.append(correct_centers)

    out_centers = torch.cat(out_centers)
    width, height = F._get_image_size(src_img)
    out_centers[:, 0] /= width
    out_centers[:, 1] /= height

    spine_plot_centers(src_img, out_centers)