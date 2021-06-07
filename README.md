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

**`loss_binary_labels` (weight 1)**
Given the pairs, the [binary cross-entropy](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.binary_cross_entropy "binary cross-entropy") is used to calculate the loss between the predictions and the ground truth. This loss replaces the `loss_labels` function which uses the classic cross-entropy loss. 

**`loss_centers` (weight 2)**
To compute the loss the [mean squared error](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.mse_loss " mean squared error") between the predictions and the centers is computed. This loss replaces the `loss_giou` function which calculates the Generalized Intersection Over Union between the bounding boxes predicted by the DETR and the ground truth. Since we don't have the bounding boxes anymore we used the MSE between the centers instead.

**`loss_spine_l1` (weight 5)**
To compute the loss the [L1 loss](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.mse_loss "L1 loss") the predictions and the centers is computed. This function replaces the `loss_boxes` function. In that case, the two losses are very similar. Instead of applying the L1 loss over the four coordinates of the bounding boxes [cx, cy, h, w], the loss is applied only among the centers [cx, cy].

### Data augmentation

### Docker

### Train

### Evaluation