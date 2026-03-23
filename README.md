# Brain Lesion Segmentation (ONNX C++ Pipeline)

Lightweight C++ inference pipeline for 3D brain lesion segmentation from multi-modal MRI volumes. Consumes a 3D U-Net ONNX model trained on BraTS 2020 and produces voxel-level segmentation masks in NIfTI format, preserving the original spatial metadata.

Designed for deployment in neuroradiology workflows where Python overhead is undesirable or unavailable.

## Model

The pipeline expects an ONNX model (opset 17) with:

- **Input**: `[batch, 4, 128, 128, 128]` corresponding to FLAIR, T1, T1ce, T2
- **Output**: `[batch, 4, 128, 128, 128]` with channels for background, NCR/NET (necrotic/non-enhancing tumor core), peritumoral edema, GD-enhancing tumor

The companion training notebook produces a 21.4 MB model. Expected Dice scores on the BraTS 2020 validation set:

| Region | Dice (Python, full val set) | Dice (C++ pipeline, 5 subjects) |
|--------|---------------------------|--------------------------------|
| NCR/NET | 0.73 | 0.71 |
| Edema (ED) | 0.85 | 0.76 |
| Enhancing Tumor (ET) | 0.84 | 0.84 |

### C++ Pipeline Validation (BraTS 2020 Training Subjects)

| Subject | NCR/NET | Edema  | Enhancing | Time (s) |
|---------|---------|--------|-----------|----------|
| 001     | 0.8563  | 0.8811 | 0.8889    | 438.6    |
| 002     | 0.8912  | 0.8251 | 0.8545    | 439.5    |
| 003     | 0.6293  | 0.7401 | 0.8513    | 434.3    |
| 004     | 0.7260  | 0.9429 | 0.8886    | 440.5    |
| 005     | 0.4682  | 0.4136 | 0.7122    | 430.1    |
| **Mean**| **0.7142**|**0.7606**|**0.8391**| **436.6**|

Mean inference time per volume: ~7.3 minutes on CPU (Intel Xeon @ 2.2GHz, single thread, ONNX Runtime 1.17.1). GPU inference with the CUDA execution provider reduces this significantly.

Place your `.onnx` file in the `models/` directory.

## Pipeline Architecture

```
NIfTI files (.nii/.nii.gz)
    |
    v
 NiftiIO::load()        Load 4 modalities, convert to float
    |
    v
 Preprocessor::run()    Z-score normalize (nonzero voxels), stack [4,D,H,W],
    |                    sliding window patch extraction with overlap
    v
 InferenceEngine        ONNX Runtime session (CPU or CUDA EP),
    |                    runs each 128^3 patch through the model
    v
 Postprocessor::run()   Average overlapping patch logits, softmax, argmax,
    |                    connected component filtering (BFS flood fill),
    v                    volume thresholding to remove spurious detections
 NiftiIO::save_labels() Write segmentation mask as NIfTI with original affine
```

Full volumes (typically 240x240x160) are processed via sliding window with configurable overlap (default 50%). Overlapping regions are averaged before the final softmax to reduce boundary artifacts.

## Dependencies

- C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.18+
- zlib (for .nii.gz decompression)
- ONNX Runtime 1.17+ (downloaded automatically by CMake via FetchContent)

No OpenCV, no ITK, no Python runtime.

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
    --output /results/BraTS20_Training_001_seg.nii.gz \
    --model models/brats_unet3d.onnx \
    --device cuda \
    --patch-overlap 0.5 \
    --min-component-size 100
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | (required) | Directory containing the four NIfTI modality files |
| `--output` | (required) | Output segmentation mask path (.nii or .nii.gz) |
| `--model` | (required) | Path to the ONNX model file |
| `--device` | `cpu` | Execution provider, `cpu` or `cuda` |
| `--patch-overlap` | `0.5` | Fraction of overlap between adjacent patches (0.0 to 0.9) |
| `--min-component-size` | `100` | Connected components below this voxel count are removed |

## Tests

Unit tests cover normalization, patch extraction, softmax, argmax, and connected component filtering. They do not require a model or data.

```bash
cd build
ctest --output-on-failure
```

## Output Labels

The segmentation mask uses integer labels following the BraTS convention:

- 0: Background
- 1: NCR/NET (necrotic and non-enhancing tumor core)
- 2: Peritumoral edema
- 3: GD-enhancing tumor

## License

This pipeline code is provided as is. The ONNX model and training data are subject to the BraTS 2020 challenge license terms.
