SpineDETR
========

Pytorch training code for the "SpineDETR" (Temporary name). SpineDETR is an adjustment of the original DETR ([Detection Transformer](https://github.com/facebookresearch/detr "Detection Transformer")) to work with 2D spine images.
Before continuing with the reading, it is recommended to read the [DETR paper](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers "DETR paper").

**Abstract**. SpineDETR base idea is to use an Object Detector to predict the location of each vertebra in a 2D spine image. What we did is to modify an already well-known Object Detector like the Detection Transformer to work with spine images.

Since our dataset contains only the vertebra centers, we decide to change the network architecture to return only them. Furthermore, since we have only one object class to predict, we changed the architecture to predict only one score for each object query. This score represents the confidence of the object query to be a vertebra or not.

Using our internal dataset of 159 images we achieved these results with a 4 fold cross validation:

| Cross Validation | Training set | Test set | TP | FN | FP | Avg error |
| :------------: | :------------: | :------------: |:------------: | :------------: |:------------: | :------------: |
| Dataset 0 | 119 | 40 | 96.44% | 3.56% | 1.86% | 2.909 mm |
| Dataset 1 | 119 | 40 | 91.38% | 8.62% | 7.27% | 3.833 mm |
| Dataset 2 | 119 | 40 | 91.75% | 8.25% | 6.59% | 3.479 mm |
| Dataset 3 | 120 | 39 | 95.75% | 4.25% | 2.88% | 3.114 mm |

Where:
- **True Positive** if the network predict a vertebra center closer than 10 mm from the closest ground truth.
- **False Negative** if the network don't predict any point for a vertebra or it is not in the radius of 10 mm.
- **False Positive** if the network predict a vertebra which cannot be assigned to any ground truth.
- **Average Error** is the error calculated on the true positive predictions.

## Architecture

## Losses

## Data augmentation

## Docker

## Train

## Evaluation