# Training Details

## Dataset

BraTS 2020 (Brain Tumor Segmentation Challenge). 369 training cases, each with four MRI modalities (FLAIR, T1, T1ce, T2) and voxel-level ground truth annotations for three tumor subregions: necrotic/non-enhancing tumor core (NCR/NET, label 1), peritumoral edema (ED, label 2), and GD-enhancing tumor (ET, label 4, remapped to 3 during training).

All volumes are skull-stripped and co-registered to a common anatomical template at 1mm isotropic resolution, with dimensions 240x240x160.

## Architecture

3D U-Net with four encoder and four decoder stages. Each stage uses two 3x3x3 convolutions with instance normalization and LeakyReLU (slope 0.01). Skip connections via concatenation. Downsampling with 2x2x2 strided convolutions, upsampling with trilinear interpolation followed by 1x1x1 convolution. Final layer is a 1x1x1 convolution projecting to 4 output channels (background + 3 tumor classes) with no activation (raw logits).

Total parameters: 5.6M.

## Preprocessing (training time)

1. Per-volume z-score normalization on nonzero voxels for each modality independently.
2. Random cropping to 128x128x128 patches.
3. Data augmentation: random flips along all three axes, random intensity scaling (0.9 to 1.1), random intensity shift (-0.1 to 0.1).

## Training

- Framework: PyTorch 2.1
- Loss: equally weighted sum of Dice loss and cross-entropy loss, computed per-class and averaged
- Optimizer: Adam, initial learning rate 1e-4
- Scheduler: cosine annealing over full training run, minimum LR 1e-6
- Batch size: 2
- Epochs: 200
- Mixed precision training via torch.cuda.amp (float16 forward pass, float32 weight updates)
- Hardware: single NVIDIA A100 40GB

## Validation Scores

Five-fold cross-validation on the BraTS 2020 training set. Reported as mean Dice across folds:

| Region | Dice |
|--------|------|
| NCR/NET | 0.73 |
| Edema (ED) | 0.85 |
| Enhancing Tumor (ET) | 0.84 |

## ONNX Export

```python
import torch

dummy = torch.randn(1, 4, 128, 128, 128, device="cuda")
torch.onnx.export(
    model,
    dummy,
    "brats_unet3d.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

Exported model size: 21.4 MB. Opset 17 is required for the instance normalization operator representation used by this architecture.

## Colab Notebook

The full training pipeline (data loading, training loop, evaluation, export) is available as a Colab notebook:

`<TODO: insert Colab notebook link here>`
