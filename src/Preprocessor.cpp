#include "Preprocessor.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

Preprocessor::Preprocessor(int patch_size, float overlap)
    : patch_sz(patch_size), overlap_frac(overlap) {}

void Preprocessor::zscore_normalize(NiftiVolume& vol) {
    double sum = 0.0;
    double sum_sq = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < vol.data.size(); i++) {
        float v = vol.data[i];
        if (v != 0.0f) {
            sum += v;
            sum_sq += static_cast<double>(v) * v;
            count++;
        }
    }

    if (count < 2) return;

    double mean = sum / static_cast<double>(count);
    double variance = (sum_sq / static_cast<double>(count)) - (mean * mean);
    double std_dev = std::sqrt(std::max(variance, 1e-8));

    for (size_t i = 0; i < vol.data.size(); i++) {
        if (vol.data[i] != 0.0f) {
            vol.data[i] = static_cast<float>((vol.data[i] - mean) / std_dev);
        }
    }
}

std::vector<float> Preprocessor::stack_modalities(const std::array<NiftiVolume, 4>& vols) {
    int nx = vols[0].nx, ny = vols[0].ny, nz = vols[0].nz;
    for (size_t c = 1; c < 4; c++) {
        if (vols[c].nx != nx || vols[c].ny != ny || vols[c].nz != nz)
            throw std::runtime_error("Modality dimension mismatch at channel " + std::to_string(c));
    }

    size_t vol_size = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    std::vector<float> stacked(4 * vol_size);

    for (size_t c = 0; c < 4; c++) {
        std::copy(vols[c].data.begin(), vols[c].data.end(),
                  stacked.begin() + static_cast<ptrdiff_t>(c * vol_size));
    }

    return stacked;
}

PatchGrid Preprocessor::extract_patches(const std::vector<float>& stacked,
                                         int channels, int D, int H, int W)
{
    int step = std::max(1, static_cast<int>(static_cast<float>(patch_sz) * (1.0f - overlap_frac)));

    auto pad_dim = [&](int orig) -> int {
        if (orig <= patch_sz) return patch_sz;
        int n_patches = (orig - patch_sz + step - 1) / step + 1;
        return (n_patches - 1) * step + patch_sz;
    };

    int pD = pad_dim(D);
    int pH = pad_dim(H);
    int pW = pad_dim(W);

    size_t sD = static_cast<size_t>(pD);
    size_t sH = static_cast<size_t>(pH);
    size_t sW = static_cast<size_t>(pW);

    std::vector<float> padded(static_cast<size_t>(channels) * sD * sH * sW, 0.0f);

    for (int c = 0; c < channels; c++) {
        for (int z = 0; z < D; z++) {
            for (int y = 0; y < H; y++) {
                const float* src_row = &stacked[static_cast<size_t>(c) * static_cast<size_t>(D) * static_cast<size_t>(H) * static_cast<size_t>(W)
                                               + static_cast<size_t>(z) * static_cast<size_t>(H) * static_cast<size_t>(W)
                                               + static_cast<size_t>(y) * static_cast<size_t>(W)];
                float* dst_row = &padded[static_cast<size_t>(c) * sD * sH * sW
                                        + static_cast<size_t>(z) * sH * sW
                                        + static_cast<size_t>(y) * sW];
                std::copy(src_row, src_row + W, dst_row);
            }
        }
    }

    PatchGrid grid;
    grid.patch_d = patch_sz;
    grid.patch_h = patch_sz;
    grid.patch_w = patch_sz;
    grid.num_channels = channels;
    grid.vol_d = D;
    grid.vol_h = H;
    grid.vol_w = W;
    grid.padded_d = pD;
    grid.padded_h = pH;
    grid.padded_w = pW;

    size_t P = static_cast<size_t>(patch_sz);
    size_t patch_vol = static_cast<size_t>(channels) * P * P * P;

    for (int z = 0; z <= pD - patch_sz; z += step) {
        for (int y = 0; y <= pH - patch_sz; y += step) {
            for (int x = 0; x <= pW - patch_sz; x += step) {
                Patch p;
                p.data.resize(patch_vol);
                p.origin_x = x;
                p.origin_y = y;
                p.origin_z = z;

                for (int c = 0; c < channels; c++) {
                    for (int dz = 0; dz < patch_sz; dz++) {
                        for (int dy = 0; dy < patch_sz; dy++) {
                            size_t src_off = static_cast<size_t>(c) * sD * sH * sW
                                           + static_cast<size_t>(z + dz) * sH * sW
                                           + static_cast<size_t>(y + dy) * sW
                                           + static_cast<size_t>(x);
                            size_t dst_off = static_cast<size_t>(c) * P * P * P
                                           + static_cast<size_t>(dz) * P * P
                                           + static_cast<size_t>(dy) * P;
                            std::copy(&padded[src_off], &padded[src_off + P],
                                      &p.data[dst_off]);
                        }
                    }
                }

                grid.patches.push_back(std::move(p));
            }
        }
    }

    return grid;
}

PatchGrid Preprocessor::run(const std::array<NiftiVolume, 4>& modalities) {
    auto vols = modalities;

    for (auto& v : vols)
        zscore_normalize(v);

    auto stacked = stack_modalities(vols);

    int D = vols[0].nz;
    int H = vols[0].ny;
    int W = vols[0].nx;

    return extract_patches(stacked, 4, D, H, W);
}
