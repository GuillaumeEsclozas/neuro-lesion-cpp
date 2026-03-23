# neuro-lesion-cpp

C++ inference pipeline for 3D brain lesion segmentation from multi-modal MRI. Takes four NIfTI volumes (FLAIR, T1, T1ce, T2) and an ONNX model, produces a segmentation mask in NIfTI format. Built for BraTS 2020 conventions.

![Segmentation example](segmentation_example.png)

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
