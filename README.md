# TAUFE: Task-Agnostic Undesirable Feature DeactivationUsing Out-of-Distribution Data

> __Publication__ </br>
> Park, D., Song, H., Kim, M., and Lee, J., "Task-Agnostic Undesirable Feature DeactivationUsing Out-of-Distribution Data," *In Proceedings of the 35th NeurIPS*, December 2021, Virtual. [[Paper]](https://openreview.net/pdf?id=4orlVaC95Bo)

## Citation
```
@article{park2021task,
  title={Task-Agnostic Undesirable Feature Deactivation Using Out-of-Distribution Data},
  author={Park, Dongmin and Song, Hwanjun and Kim, MinSeok and Lee, Jae-Gil},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## 1. Overview
A deep neural network (DNN) has achieved great success in many machine learning tasks by virtue of its high expressive power. However, its prediction can be easily biased to undesirable features, which are not essential for solving the target task and are even imperceptible to a human, thereby resulting in poor generalization. Leveraging plenty of undesirable features in out-of-distribution (OOD) examples has emerged as a potential solution for de-biasing such features, and a recent study shows that softmax-level calibration of OOD examples can successfully remove the contribution of undesirable features to the last fully-connected layer of a classifier. However, its applicability is confined to the classification task, and its impact on a DNN feature extractor is not properly investigated. In this paper, we propose Taufe, a novel regularizer that deactivates many undesirable features using OOD examples in the feature extraction layer and thus removes the dependency on the task-specific softmax layer. To show the task-agnostic nature of Taufe, we rigorously validate its performance on three tasks, classification, regression, and a mix of them, on CIFAR-10, CIFAR-100, ImageNet, CUB200, and CAR datasets. The results demonstrate that Taufe consistently outperforms the state-of-the-art method as well as the baselines without regularization. 


## 2. How to run
1. Image classification task
- go to the folder 'classification/', and run STANDARD.py, TAUFE.py with arguments:
```
--in-data-name: the name of a target in-distribution dataset (string) # cifar10, cifar100, imgnet10
--ood-data-name: the name of an out-of-distribution dataset (string) # lsun, 80mTiny, svhn, imgnet990, places365
--n-samples: the number of training samples for few-shot learning (integer)
--n-class: the number of classes (int)
--taufe-weight: hyper-paramter lambda for taufe loss (float) # default:0.1
```

2. Semi-supervised learning task
- go to the folder 'SSL/', and run MixMatch.py with arguments:
```
--in-data-name: the name of a target in-distribution dataset (string) # cifar10, cifar100
--ood-data-name: the name of an out-of-distribution dataset (string) # lsun, 80mTiny, svhn
--n-labeled: the number of labeled samples (integer)
--train-iteration: the number of training iterations (int)
--taufe-weight: hyper-paramter lambda for taufe loss (float) # default:0.1
```

3. Bounding-box regression task
- go to the folder 'regression/', and run bbox_Standard.py, bbox_TAUFE.py with arguments:
```
--in-data-name: the name of a target in-distribution dataset (string) # cub200, car
--ood-data-name: the name of an out-of-distribution dataset (string) # imgnet, places365
--loss-type: the name of loss type (string) # L1, L1-IoU, D-IoU
--n-class: the number of classes (int)
--n-shots: the number of samples per class (int)
--taufe-weight: hyper-paramter lambda for taufe loss (float) # default:0.1
```


##  3. Requirement 
- Python 3
- torch >= 1.3.0
