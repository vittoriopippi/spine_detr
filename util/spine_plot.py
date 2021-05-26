import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

def spine_plot_centers(img, output, color=(255, 0, 0), threshold=None, logits=None):
    if isinstance(img, torch.Tensor):
        img = F.to_pil_image(torch.clone(img))
    assert isinstance(img, Image.Image)
    output = torch.clone(output)
    # img = img.convert('LA').convert('RGB')
    output = to_pixel(img, output)

    if threshold is not None and logits is not None:
        output = output[logits.squeeze(-1) > threshold]

    r = 3
    draw = ImageDraw.Draw(img)
    for x, y in output:
        x, y = x.item(), y.item()
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    return img


def spine_to_pil(src, to_grayscale=True):
    img = F.to_pil_image(torch.clone(src))
    return img.convert('LA').convert('RGB')

def to_pixel(img, points):
    width, height = F._get_image_size(img)
    points[:, 0] *= width
    points[:, 1] *= height
    return points.long()

def spine_plot_connection(img, targets, predicted):
    assert isinstance(img, Image.Image)
    targets = torch.clone(targets)
    predicted = torch.clone(predicted)

    targets = to_pixel(img, targets)
    predicted = to_pixel(img, predicted)

    r = 3
    draw = ImageDraw.Draw(img)
    for t, p in zip(targets, predicted):
        tx, ty = t
        px, py = p
        tx, ty, px, py = tx.item(), ty.item(), px.item(), py.item()
        draw.line((tx, ty, px, py), fill=(255, 255, 255))
        draw.ellipse((tx-r, ty-r, tx+r, ty+r), fill=(0, 255, 0))
        draw.ellipse((px-r, py-r, px+r, py+r), fill=(0, 0, 255))

    return img


def spine_class(img, predicted):
    assert isinstance(img, Image.Image)
    predicted = torch.clone(predicted)
    predicted = torch.clamp(predicted, 0, 1)

    w, h = F._get_image_size(img)

    rect_w = h // len(predicted)
    rect_h = h // len(predicted)

    draw = ImageDraw.Draw(img)
    # print_column(predicted[:, 0], draw, w - (rect_w * 2), rect_w, rect_h)
    print_column(predicted[:, 0], draw, w - rect_w, rect_w, rect_h)
    return img

def print_column(predicted, draw, x_start, rect_w, rect_h):
    for i, p in enumerate(predicted):
        x0, y0, x1, y1 = x_start, i * rect_h, x_start + rect_w, (i + 1) * rect_h
        val = int(255 * p.item())
        color = (255-val, val, 0)
        draw.rectangle([x0, y0, x1, y1], fill=color)





