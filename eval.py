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
import pandas as pd
import os
import json

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

def batch_gen(img_path, stride=None, max_batch_size=32):
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

        if len(images) >= max_batch_size:
            batch = torch.stack(images)
            yield batch, wh_list, image
            images = []

    if len(images) > 0:
        batch = torch.stack(images)
        return batch, wh_list, image

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for cval in range(4):
        args.resume = args.resume.format(cval)
        model = SpineDETR(args)
        df = pd.read_csv(f'fake_coco/csv_test_{cval}.csv')
        df = df[['patient_id', 'cross_val', 'filename']].drop_duplicates()

        for row_i, (index, row) in enumerate(df.iterrows()):
            img_path = args.spine_folder + f"{row['patient_id']}/{row['filename']}"

            all_logits, all_centers = [], []
            for batch, windows, src_img in batch_gen(img_path, stride=args.stride, max_batch_size=args.batch_size):
                out = model(batch)
                all_logits.append(out['pred_logits'])
                all_centers.append(out['pred_boxes'])

            all_logits = torch.cat(all_logits)
            all_centers = torch.cat(all_centers)

            window_size = args.rand_crop
            window_centers = torch.Tensor(windows).to(args.device)
            window_centers += window_size // 2

            out_centers = []
            for i, (window, logits, centers) in enumerate(zip(windows, all_logits, all_centers)):
                centers = centers[logits.squeeze() > 0.5]
                centers = centers * window_size
                centers[:, 0] += window[0]
                centers[:, 1] += window[1]
                if centers.numel() > 0:
                    dist = torch.cdist(centers, window_centers)
                    min_idx = dist.argmin(1)
                    correct_centers = centers[min_idx == i]
                    out_centers.append(correct_centers)

            out_centers = torch.cat(out_centers)
            os.makedirs(f'{args.output_dir}/{row["patient_id"]}', exist_ok=True)

            sorted_idx = torch.argsort(out_centers[:, 1])
            out_centers = out_centers[sorted_idx]

            out_dict = {}
            for i, (x, y) in enumerate(out_centers):
                out_dict[str(i)] = [{'pos': [x.item(), y.item()]}]
            with open(f'{args.output_dir}/{row["patient_id"]}/{row["patient_id"]}.json', 'w') as f:
                json.dump(out_dict, f)

            width, height = F._get_image_size(src_img)
            out_centers[:, 0] /= width
            out_centers[:, 1] /= height

            out_image = spine_plot_centers(src_img, out_centers)
            out_image.save(f'{args.output_dir}/{row["patient_id"]}/{row["patient_id"]}.jpg')
            
            print(f'  CVAL: {cval} progress: {row_i}/{len(df)}{" " * 10}', end='\r')

