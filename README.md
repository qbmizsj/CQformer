# CQ-former

This project implements a segmentation model that works with various medical imaging datasets.

## Requirements

Code developed and tested in Python 3.8.12 using PyTorch 1.10.0. Please refer to their official websites for installation and setup.

### Major Requirements

```plaintext
nibabel @ file:///home/conda/feedstock_root/build_artifacts/nibabel_1673318073381/work
numpy==1.24.2
pandas==1.5.3
torchio==0.18.90
torchvision==0.14.1+cu116
monai==1.3.0 
```

## Datasets

To run the model, download the following datasets:

1. **Heart and Prostate**: [Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
2. **Abdomen**: BTCV Challenge dataset can be found at [Synapse](https://www.synapse.org/Synapse:syn3193805/wiki/217752)

## Running the Model

To run the model in the background using `nohup`, use the following command:

```bash
nohup python s2s_main.py --name <dataset_name> --root <source_path> --model_lr <learning_rate> --batch_size <size> --group <depth> --epochs <num_epochs> --in_channel <channels> --num_classes <classes> --hidden_size <hidden_dim> --img_size <img_size> --size <volume_size> --n_skip <skip_connect> --vit_name <vit_model> --vit_patches_size <patch_size> --optim <optimizer> --result_path <save_path> --seed <seed_value> &
```

### Argument Description

- `--name`: Name of the dataset (e.g., Heart, Prostate).
- `--root`: Source path of the dataset.
- `--model_lr`: Learning rate of the model (default: 0.01).
- `--batch_size`: Size of each batch (default: 8).
- `--group`: Depth of the model (default: 8).
- `--epochs`: Number of training epochs (default: 300).
- `--in_channel`: Number of image modalities (default: 1).
- `--num_classes`: Number of label classes (default: 8).
- `--hidden_size`: Dimension of hidden state (default: 768).
- `--img_size`: Input image size (default: (128, 128)).
- `--size`: Input volume size (default: (128, 128, 8)).
- `--n_skip`: Number of skip connections (default: 3).
- `--vit_name`: Selected ViT model (default: R50-ViT-B_16).
- `--vit_patches_size`: Size of ViT patches (default: 16).
- `--optim`: Type of optimizer (default: sgd).
- `--result_path`: Path to save results.
- `--seed`: Random seed (default: 44).

## Questions

If you have any questions about the implementation of CQ-former or data preprocessing, please contact me at:

```plaintext
xx@gmail.com
```
