#pragma once

#include "NiftiIO.h"
#include <vector>
#include <array>

/*
 * A patch is a dense [C, D, H, W] tensor stored flat, plus the (x, y, z)
 * corner in the original volume so we know where to put results back.
 */
struct Patch {
    std::vector<float> data;   /* C * pD * pH * pW floats */
    int origin_x, origin_y, origin_z;
};

struct PatchGrid {
    std::vector<Patch> patches;
    int patch_d, patch_h, patch_w;
    int num_channels;
    /* original volume dimensions (before any padding) */
    int vol_d, vol_h, vol_w;
    /* padded volume dimensions */
    int padded_d, padded_h, padded_w;
};

class Preprocessor {
public:
    explicit Preprocessor(int patch_size = 128, float overlap = 0.5f);

    /*
     * Full preprocessing: takes 4 modality volumes, normalizes each (z-score
     * on nonzero voxels), stacks them into a [4, D, H, W] tensor, then
     * extracts sliding-window patches with the configured overlap.
     */
    PatchGrid run(const std::array<NiftiVolume, 4>& modalities);

    /* Normalize a single volume in-place: z-score on nonzero voxels. */
    static void zscore_normalize(NiftiVolume& vol);

    /* Stack 4 volumes into a flat [4, D, H, W] buffer. All must have same dims. */
    static std::vector<float> stack_modalities(const std::array<NiftiVolume, 4>& vols);

    /* Extract patches from a [C, D, H, W] tensor. */
    PatchGrid extract_patches(const std::vector<float>& stacked,
                              int channels, int D, int H, int W);

private:
    int patch_sz;
    float overlap_frac;
};
