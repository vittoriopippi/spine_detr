SpineDETR
========

Pytorch training code for the "SpineDETR" (Temporary name). SpineDETR is an adjustment of the original DETR ([Detection Transformer](https://github.com/facebookresearch/detr "Detection Transformer")) to work with 2D spine images.
Before continuing with the reading, it is recommended to read the [DETR paper](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers "DETR paper").

**Abstract**. SpineDETR base idea is to use an Object Detector to predict the location of each vertebra in a 2D spine image. We modified an already well-known Object Detector like the Detection Transformer to work with spine images.

Since our dataset contains only the vertebra centers, we decide to change the network architecture to return only them. Furthermore, since we have only one object class to predict, we changed the architecture to predict only one score for each object query. This score represents the confidence of the object query to be a vertebra or not.


------------



Using our internal dataset of 159 images we achieved these results with a 4 fold cross-validation:

| Cross Validation | Training set | Test set | TP | FN | FP | Avg error |
| :------------: | :------------: | :------------: |:------------: | :------------: |:------------: | :------------: |
| Dataset 0 | 119 | 40 | 96.44% | 3.56% | 1.86% | 2.909 mm |
| Dataset 1 | 119 | 40 | 91.38% | 8.62% | 7.27% | 3.833 mm |
| Dataset 2 | 119 | 40 | 91.75% | 8.25% | 6.59% | 3.479 mm |
| Dataset 3 | 120 | 39 | 95.75% | 4.25% | 2.88% | 3.114 mm |

During the evaluation, we combine the network predictions with the ground truth. Given N the vertebrae predicted by the network and M the ground truth, we apply the Hungarian algorithm to assign each prediction to the ground truth and minimize the overall distance.

Where for each prediction we refer as:
- **True Positive** if the vertebra center is closer than 10 mm from the assigned ground truth.
- **False Negative** if the vertebra center is further than 10 mm from the assigned ground truth.
- **False Positive** if the network predicts a vertebra that is not assigned to any ground truth.
- **Average Error** is the error calculated among all true positives.

### Architecture
The original DETR architecture remains untouched, the only different parts are the FFNs (Fast Forward Networks), which are slightly changed to predict vertebrae centers and the confidence scores.

Given N the number of object queries used, one network predicts the vertebrae centers with shape [N, 2] and another network predicts the confidence score with shape [N, 1].

Like the original DETR, the vertebra centers are represented as the couple [cx, cy]. Since there is only one class, instead of using the softmax, we modified the network to predict only one score in the range [0,1]. Each object query has a confidence score which tells us how much likely that query is a vertebra or not.

### Losses
Since the network outputs are changed, we had to modify also the losses used. 

Like DETR, during the loss computations, the best assignment between the predictions and the ground truth is used. The assignment is made with the Hungarian algorithm.

The following losses are combined with a weighted sum. Each one of them is a replacement for the loss used in DETR.

###### **[`loss_binary_labels`](https://github.com/vittoriopippi/ostfalia_detr/blob/d952a0e18ba7fc91c27e7332cec19e75fa1b7a3e/models/detr.py#L138) (weight 1)**
Given the pairs, the [binary cross-entropy](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.binary_cross_entropy "binary cross-entropy") is used to calculate the loss between the predictions and the ground truth. This loss replaces the `loss_labels` function which uses the classic cross-entropy loss. 

###### **[`loss_centers`](https://github.com/vittoriopippi/ostfalia_detr/blob/d952a0e18ba7fc91c27e7332cec19e75fa1b7a3e/models/detr.py#L196) (weight 2)**
To compute the loss the [mean squared error](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.mse_loss " mean squared error") between the predictions and the centers is computed. This loss replaces the `loss_giou` function which calculates the Generalized Intersection Over Union between the bounding boxes predicted by the DETR and the ground truth. Since we don't have the bounding boxes anymore we used the MSE between the centers instead.

###### **[`loss_spine_l1`](https://github.com/vittoriopippi/ostfalia_detr/blob/d952a0e18ba7fc91c27e7332cec19e75fa1b7a3e/models/detr.py#L209) (weight 5)**
To compute the loss the [L1 loss](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.mse_loss "L1 loss") the predictions and the centers is computed. This function replaces the `loss_boxes` function. In that case, the two losses are very similar. Instead of applying the L1 loss over the four coordinates of the bounding boxes [cx, cy, h, w], the loss is applied only among the centers [cx, cy].

### Data augmentation
Inside the script "[`/dataset/spine_2d.py`](https://github.com/vittoriopippi/ostfalia_detr/blob/d952a0e18ba7fc91c27e7332cec19e75fa1b7a3e/datasets/spine_2d.py#L78 "/dataset/spine_2d.py")" you can find the transformations applied during training and test.

```python
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
```

All custom transformations are defined inside "[`/dataset/spine_transforms.py`](https://github.com/vittoriopippi/ostfalia_detr/blob/master/datasets/spine_transforms.py "/dataset/spine_transforms.py")". For each transformation, the parent is the correspondent transformation available in PyTorch. The `forward` and `__call__` methods have been modified to applying the filter also to the ground truth centers.

[`FakeTransform`](https://github.com/vittoriopippi/ostfalia_detr/blob/d952a0e18ba7fc91c27e7332cec19e75fa1b7a3e/datasets/spine_transforms.py#L13 "FakeTransform class definition") is just a placeholder transformation that doesn't do anything.

Each transform gets as input not only an image but a `sample`. Each `sample` is a dictionary with three keys:
```python
sample = {
    'image': PIL.Image.Image
    'vertebrae': torch.Tensor
    'info': {
        'spacing': numpy.float64
        'patient_id': int
        'src_width': int
        'src_height': int
    }
}
```

`sample['vertebrae']` is a tensor with shape [N, 4] where N is the number of vertebrae in the image. As defined [here](https://github.com/vittoriopippi/ostfalia_detr/blob/d952a0e18ba7fc91c27e7332cec19e75fa1b7a3e/datasets/spine_2d.py#L49 "/dataset/spine_2d.py"), each vertebra has 4 values:  `vertebrae_id`, `center_x`, `center_y`, `genant_score`. Only the center coordinates are used so far.
The `vertebrae_id` defines the vertebra following this map:

```python
map = {
        'c1': 0,  'c2': 1,  'c3': 2,  'c4':  3,  'c5':  4,  'c6':  5,  'c7': 6, 'c8': 7,
        't1': 8,  't2': 9,  't3': 10, 't4':  11, 't5':  12, 't6':  13,
        't7': 14, 't8': 15, 't9': 16, 't10': 17, 't11': 18, 't12': 19,
        'l1': 20, 'l2': 21, 'l3': 22, 'l4':  23, 'l5':  24,
        's1': 25, 's2': 26, 's3': 27, 's4':  28, 's5':  29,
    }
```
### Docker

### Train

### Evaluation