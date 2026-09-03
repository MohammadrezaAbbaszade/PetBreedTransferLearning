# Pet Breed Classification with Transfer Learning

Fine-grained classification of **37 cat and dog breeds** from the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) using an ImageNet-pretrained ResNet18.

The project compares frozen feature extraction, selective and full fine-tuning, BatchNorm-statistics control, Grad-CAM interpretation, and mixup regularization on a deliberately small, class-balanced dataset.

## Results

| Experiment | Trainable parameters | Best validation accuracy |
|---|---:|---:|
| Frozen backbone | 18,981 | 86.1% |
| Fine-tune `layer4` | 8,412,709 | 86.5% |
| Fine-tune `layer4`, frozen BN statistics | 8,412,709 | **87.2%** |
| Fine-tune all layers | 11,195,493 | 85.5% |
| Frozen backbone + mixup | 18,981 | 84.8% |

Selective fine-tuning produced the strongest model. Updating `layer4` while retaining ImageNet BatchNorm running statistics reached **87.2%** validation accuracy. Full fine-tuning performed worse despite exposing more parameters, which is consistent with the limited training set of only 740 images. Mixup also reduced accuracy under the tested eight-epoch schedule.

## What the notebook covers

- Balanced sampling of 20 training and 8 validation images per breed
- Frozen-backbone transfer-learning baseline
- Parameter groups with separate learning rates for the head and backbone
- Partial (`layer4`) versus full-network fine-tuning
- BatchNorm running-statistics comparison for small-data training
- Grad-CAM visualizations of frozen and fine-tuned models
- Controlled plain-versus-mixup comparison

## Dataset

| Split | Source split | Images per breed | Total |
|---|---|---:|---:|
| Training | `trainval` | 20 | 740 |
| Validation | `test` | 8 | 296 |

The subset is balanced and reproducible with seed 42. The dataset downloads automatically through `torchvision` into `data/`, which is excluded from Git.

## Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook pet_breed_transfer_learning.ipynb
```

The notebook selects CUDA when available, then Apple Silicon MPS, and otherwise uses CPU.

## Reproduce the experiments

Run the notebook from top to bottom. The first run downloads the Oxford-IIIT Pet Dataset. Training time depends on the selected device; all reported comparisons use the same split, seed, augmentation pipeline, optimizer settings, and eight-epoch budget.

## Limitations

The reported results use one random seed and a small subset of the full dataset. Grad-CAM provides qualitative evidence about spatial attention but does not prove that the model relies exclusively on breed-specific features. A stronger evaluation should repeat training across several seeds and use larger class-balanced subsets.

## References

- [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
