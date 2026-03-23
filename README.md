# neuro-lesion-cpp

Inference pipeline for 3D brain lesion segmentation from multi-modal MRI volumes. Written in C++17, uses ONNX Runtime for model inference. No Python, OpenCV or ITK dependency at runtime.

![Segmentation example](image/brain.png)

## Dependencies

- C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.18+
- zlib
- ONNX Runtime 1.17+ (fetched automatically by CMake)

## Model

The pipeline expects an ONNX model (opset 17) with input `[batch, 4, 128, 128, 128]` (FLAIR, T1, T1ce, T2) and output `[batch, 4, 128, 128, 128]` (background, NCR/NET, edema, enhancing tumor). The model is not shipped with this repository. A companion Colab notebook trains a lightweight 3D U-Net (5.6M parameters, Dice + CE loss, mixed precision, 200 epochs) on BraTS 2020 and exports to ONNX. The exported model is 21.4 MB.

## Building
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

## Usage

The four BraTS modality files must be in a single directory. The pipeline identifies them by filename substring (`_flair`, `_t1`, `_t1ce`, `_t2`).
```bash
./brain_lesion_seg \
    --input-dir /data/BraTS20_Training_001 \
    --output /results/seg_001.nii \
    --model /path/to/brats_unet3d.onnx \
    --device cpu \
    --patch-overlap 0.5 \
    --min-component-size 100
```

`--device` accepts `cpu` or `cuda`. `--patch-overlap` controls overlap between adjacent 128^3 patches. `--min-component-size` removes connected components below this voxel count.

## Pipeline

NiftiIO::load()          Load 4 modalities from NIfTI, convert to float
Preprocessor::run()      Z-score normalize nonzero voxels, sliding window patch extraction
InferenceEngine          ONNX Runtime session, run each patch
Postprocessor::run()     Average overlapping logits, softmax, argmax, BFS flood fill filtering
NiftiIO::save_labels()   Write mask as NIfTI preserving original affine

## Output labels

0 background, 1 NCR/NET, 2 peritumoral edema, 3 GD-enhancing tumor.

## Validation

Tested on 5 BraTS 2020 training subjects to verify pipeline correctness against the Python reference. For model performance on unseen data, the training notebook reports mean foreground Dice 0.7317 on the full validation set.

| Subject | NCR/NET | Edema  | Enhancing | Time (s) |
|---------|---------|--------|-----------|----------|
| 001     | 0.8563  | 0.8811 | 0.8889    | 438.6    |
| 002     | 0.8912  | 0.8251 | 0.8545    | 439.5    |
| 003     | 0.6293  | 0.7401 | 0.8513    | 434.3    |
| 004     | 0.7260  | 0.9429 | 0.8886    | 440.5    |
| 005     | 0.4682  | 0.4136 | 0.7122    | 430.1    |
| Mean    | 0.7142  | 0.7606 | 0.8391    | 436.6    |

Mean time per volume: ~7.3 min on CPU (Intel Xeon @ 2.2GHz, single thread, ONNX Runtime 1.17.1).

## Tests
```bash
cd build
./test_preprocessor
./test_postprocessor
```

## References

B.H. Menze et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS). IEEE TMI, 34(10):1993-2024, 2015.
