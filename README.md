import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np

# Load data for subject 001
subj_dir = "/content/brats20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001"
t1ce = nib.load(f"{subj_dir}/BraTS20_Training_001_t1ce.nii").get_fdata()
gt = nib.load(f"{subj_dir}/BraTS20_Training_001_seg.nii").get_fdata()
pred = nib.load("/content/output_seg_001.nii").get_fdata()

# Remap GT label 4 -> 3
gt[gt == 4] = 3

# Pick a good axial slice (where tumor is visible)
tumor_counts = [np.sum(gt[:,:,z] > 0) for z in range(gt.shape[2])]
best_slice = np.argmax(tumor_counts)

# Color map: 1=red (NCR/NET), 2=green (edema), 3=yellow (enhancing)
colors = {1: [1, 0, 0], 2: [0, 1, 0], 3: [1, 1, 0]}

def overlay(mri_slice, seg_slice, alpha=0.4):
    mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    rgb = np.stack([mri_norm]*3, axis=-1)
    for label, color in colors.items():
        mask = seg_slice == label
        for c in range(3):
            rgb[:,:,c] = np.where(mask, rgb[:,:,c]*(1-alpha) + color[c]*alpha, rgb[:,:,c])
    return rgb

fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')

axes[0].imshow(overlay(t1ce[:,:,best_slice], gt[:,:,best_slice]).T, origin='lower')
axes[0].set_title("Ground Truth", color='white', fontsize=12)
axes[0].axis('off')

axes[1].imshow(overlay(t1ce[:,:,best_slice], pred[:,:,best_slice]).T, origin='lower')
axes[1].set_title("C++ Pipeline Output", color='white', fontsize=12)
axes[1].axis('off')

legend_patches = [
    mpatches.Patch(color=[1,0,0], label='NCR/NET'),
    mpatches.Patch(color=[0,1,0], label='Edema'),
    mpatches.Patch(color=[1,1,0], label='Enhancing'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           facecolor='black', edgecolor='white', labelcolor='white', fontsize=10)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("/content/neuro-lesion-cpp/segmentation_example.png", dpi=150, bbox_inches='tight',
            facecolor='black', edgecolor='none')
plt.show()
print("Saved to /content/neuro-lesion-cpp/segmentation_example.png")
```

Une fois la figure générée, télécharge la (`segmentation_example.png`), puis upload la sur ton repo GitHub: va dans le repo → Add file → Upload files → drop le PNG → commit.

Ensuite, remplace tout le README.md sur GitHub (edit dans le browser) par ceci:
```
# neuro-lesion-cpp

C++ inference pipeline for 3D brain lesion segmentation from multi-modal MRI. Takes four NIfTI volumes (FLAIR, T1, T1ce, T2) and an ONNX model, produces a segmentation mask in NIfTI format. Built for BraTS 2020 conventions.

![Segmentation example](brain.png)

## Dependencies

This code requires C++17, CMake 3.18+ and zlib. ONNX Runtime 1.17+ is fetched automatically by CMake via FetchContent. No OpenCV, no ITK, no Python runtime required at inference time.

The ONNX model is not included. You need to provide your own (see the `training/` directory for instructions on how to produce one from BraTS 2020 data using PyTorch).

## Model expectations

The pipeline expects an ONNX model (opset 17) with input shape `[batch, 4, 128, 128, 128]` (FLAIR, T1, T1ce, T2) and output shape `[batch, 4, 128, 128, 128]` (background, NCR/NET, edema, enhancing tumor). The companion training notebook produces a 21.4 MB model with 5.6M parameters trained for 200 epochs with Dice + cross entropy loss and mixed precision.

## Building
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

On Windows with Visual Studio:
```bash
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Running

Place the four BraTS modality files in a single directory. The pipeline identifies them by filename (looks for `_flair`, `_t1`, `_t1ce`, `_t2` substrings).
```bash
./brain_lesion_seg \
    --input-dir /data/BraTS20_Training_001 \
    --output /results/seg_001.nii \
    --model models/brats_unet3d.onnx \
    --device cpu \
    --patch-overlap 0.5 \
    --min-component-size 100
```

`--device` accepts `cpu` or `cuda`. `--patch-overlap` controls the fraction of overlap between adjacent 128^3 patches (default 0.5). `--min-component-size` removes connected components smaller than this many voxels (default 100).

Full volumes (typically 240x240x160) are processed via sliding window. Overlapping regions are averaged before softmax to reduce boundary artifacts.

## Pipeline
```
NIfTI files (.nii/.nii.gz)
    |
    v
NiftiIO::load()           Load 4 modalities, convert to float
    |
    v
Preprocessor::run()       Z-score normalize (nonzero voxels), stack to [4,D,H,W],
    |                     sliding window patch extraction with overlap
    v
InferenceEngine           ONNX Runtime session (CPU or CUDA),
    |                     run each 128^3 patch through the model
    v
Postprocessor::run()      Average overlapping logits, softmax, argmax,
    |                     connected component filtering (BFS flood fill),
    v                     volume thresholding
NiftiIO::save_labels()    Write segmentation mask preserving original affine
```

## Output labels

Following BraTS conventions: 0 background, 1 NCR/NET (necrotic and non-enhancing tumor core), 2 peritumoral edema, 3 GD-enhancing tumor.

## Validation

Tested on 5 BraTS 2020 training subjects. The model was validated on the full BraTS validation set in Python (see `training/`).

| Subject | NCR/NET | Edema  | Enhancing | Time (s) |
|---------|---------|--------|-----------|----------|
| 001     | 0.8563  | 0.8811 | 0.8889    | 438.6    |
| 002     | 0.8912  | 0.8251 | 0.8545    | 439.5    |
| 003     | 0.6293  | 0.7401 | 0.8513    | 434.3    |
| 004     | 0.7260  | 0.9429 | 0.8886    | 440.5    |
| 005     | 0.4682  | 0.4136 | 0.7122    | 430.1    |
| Mean    | 0.7142  | 0.7606 | 0.8391    | 436.6    |

Mean inference time per volume: ~7.3 minutes on a single CPU thread (Intel Xeon @ 2.2GHz, ONNX Runtime 1.17.1). CUDA execution provider reduces this significantly.

## Tests

Unit tests cover normalization edge cases, patch extraction at volume boundaries, softmax, argmax, and connected component filtering. No model or data needed.
```bash
cd build
ctest --output-on-failure
```

## References

The training data comes from the BraTS 2020 challenge. If you use this pipeline or the companion model, please cite:

B.H. Menze et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS). IEEE TMI, 34(10):1993-2024, 2015.

## License

Pipeline code is provided as is. The ONNX model and training data are subject to the BraTS 2020 challenge license terms.
