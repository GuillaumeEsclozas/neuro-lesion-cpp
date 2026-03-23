#pragma once

#include "Preprocessor.h"
#include <vector>

class Postprocessor {
public:
    explicit Postprocessor(int min_component_voxels = 100);

    // Aggregate overlapping patches -> softmax -> argmax -> remove small blobs.
    // Returns D*H*W label map.

    std::vector<int> run(const std::vector<std::vector<float>>& patch_logits,
                         const PatchGrid& grid);

    /* Aggregate overlapping patches by averaging. Returns [C, D, H, W] logits. */
    static std::vector<float> aggregate_patches(
        const std::vector<std::vector<float>>& patch_logits,
        const PatchGrid& grid);

    /* In-place softmax across channel dim of a [C, D, H, W] tensor. */
    static void softmax_channels(std::vector<float>& probs, int C, int D, int H, int W);

    /* Argmax across channel dim. Returns [D*H*W] int labels. */
    static std::vector<int> argmax(const std::vector<float>& probs, int C, int D, int H, int W);

    /* Remove connected components smaller than threshold voxels.
     * Operates per nonzero label. Uses BFS flood fill. */
    static void filter_small_components(std::vector<int>& labels, int D, int H, int W,
                                         int min_size);

private:
    int min_component_sz;
};
