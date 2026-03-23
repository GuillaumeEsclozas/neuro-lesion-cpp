#include "Postprocessor.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <numeric>

Postprocessor::Postprocessor(int min_component_voxels)
    : min_component_sz(min_component_voxels) {}

std::vector<float> Postprocessor::aggregate_patches(
    const std::vector<std::vector<float>>& patch_logits,
    const PatchGrid& grid)
{
    size_t C = static_cast<size_t>(grid.num_channels);
    size_t pD = static_cast<size_t>(grid.padded_d);
    size_t pH = static_cast<size_t>(grid.padded_h);
    size_t pW = static_cast<size_t>(grid.padded_w);
    size_t P = static_cast<size_t>(grid.patch_d);
    size_t vol_total = C * pD * pH * pW;

    std::vector<float> accum(vol_total, 0.0f);
    std::vector<float> counts(pD * pH * pW, 0.0f);

    for (size_t pi = 0; pi < grid.patches.size(); pi++) {
        const Patch& patch = grid.patches[pi];
        const auto& logits = patch_logits[pi];
        size_t ox = static_cast<size_t>(patch.origin_x);
        size_t oy = static_cast<size_t>(patch.origin_y);
        size_t oz = static_cast<size_t>(patch.origin_z);

        for (size_t c = 0; c < C; c++) {
            for (size_t dz = 0; dz < P; dz++) {
                for (size_t dy = 0; dy < P; dy++) {
                    for (size_t dx = 0; dx < P; dx++) {
                        size_t src_idx = c * P * P * P
                                       + dz * P * P
                                       + dy * P + dx;

                        size_t gz = oz + dz, gy = oy + dy, gx = ox + dx;
                        size_t dst_idx = c * pD * pH * pW
                                       + gz * pH * pW
                                       + gy * pW + gx;

                        accum[dst_idx] += logits[src_idx];

                        if (c == 0) {
                            size_t cnt_idx = gz * pH * pW + gy * pW + gx;
                            counts[cnt_idx] += 1.0f;
                        }
                    }
                }
            }
        }
    }

    // average over overlap count
    size_t spatial_padded = pD * pH * pW;
    for (size_t c = 0; c < C; c++) {
        for (size_t i = 0; i < spatial_padded; i++) {
            if (counts[i] > 0.0f) {
                accum[c * spatial_padded + i] /= counts[i];
            }
        }
    }

    // crop back to original dims
    size_t D = static_cast<size_t>(grid.vol_d);
    size_t H = static_cast<size_t>(grid.vol_h);
    size_t W = static_cast<size_t>(grid.vol_w);
    std::vector<float> result(C * D * H * W);

    for (size_t c = 0; c < C; c++) {
        for (size_t z = 0; z < D; z++) {
            for (size_t y = 0; y < H; y++) {
                size_t src_off = c * pD * pH * pW + z * pH * pW + y * pW;
                size_t dst_off = c * D * H * W + z * H * W + y * W;
                std::copy(&accum[src_off], &accum[src_off + W], &result[dst_off]);
            }
        }
    }

    return result;
}

void Postprocessor::softmax_channels(std::vector<float>& probs, int C, int D, int H, int W) {
    size_t spatial = static_cast<size_t>(D) * static_cast<size_t>(H) * static_cast<size_t>(W);

    for (size_t i = 0; i < spatial; i++) {
        float max_val = probs[i];
        for (int c = 1; c < C; c++) {
            float v = probs[static_cast<size_t>(c) * spatial + i];
            if (v > max_val) max_val = v;
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C; c++) {
            float e = std::exp(probs[static_cast<size_t>(c) * spatial + i] - max_val);
            probs[static_cast<size_t>(c) * spatial + i] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < C; c++) {
            probs[static_cast<size_t>(c) * spatial + i] *= inv_sum;
        }
    }
}

std::vector<int> Postprocessor::argmax(const std::vector<float>& probs, int C, int D, int H, int W) {
    size_t spatial = static_cast<size_t>(D) * static_cast<size_t>(H) * static_cast<size_t>(W);
    std::vector<int> labels(spatial);

    for (size_t i = 0; i < spatial; i++) {
        int best = 0;
        float best_val = probs[i];
        for (int c = 1; c < C; c++) {
            float v = probs[static_cast<size_t>(c) * spatial + i];
            if (v > best_val) {
                best_val = v;
                best = c;
            }
        }
        labels[i] = best;
    }

    return labels;
}

void Postprocessor::filter_small_components(std::vector<int>& labels,
                                             int D, int H, int W, int min_size)
{
    size_t total = static_cast<size_t>(D) * static_cast<size_t>(H) * static_cast<size_t>(W);
    std::vector<bool> visited(total, false);

    const int ddx[] = {-1, 1,  0, 0,  0, 0};
    const int ddy[] = { 0, 0, -1, 1,  0, 0};
    const int ddz[] = { 0, 0,  0, 0, -1, 1};

    auto idx = [&](int x, int y, int z) -> size_t {
        return static_cast<size_t>(z) * static_cast<size_t>(H) * static_cast<size_t>(W)
             + static_cast<size_t>(y) * static_cast<size_t>(W)
             + static_cast<size_t>(x);
    };

    std::queue<std::array<int,3>> bfs_queue;

    for (int z = 0; z < D; z++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                size_t i = idx(x, y, z);
                if (visited[i] || labels[i] == 0) continue;

                int lbl = labels[i];
                std::vector<size_t> component;
                bfs_queue.push({x, y, z});
                visited[i] = true;

                while (!bfs_queue.empty()) {
                    auto [cx, cy, cz] = bfs_queue.front();
                    bfs_queue.pop();
                    component.push_back(idx(cx, cy, cz));

                    for (int d = 0; d < 6; d++) {
                        int nx = cx + ddx[d], ny = cy + ddy[d], nz = cz + ddz[d];
                        if (nx < 0 || nx >= W || ny < 0 || ny >= H || nz < 0 || nz >= D)
                            continue;
                        size_t ni = idx(nx, ny, nz);
                        if (!visited[ni] && labels[ni] == lbl) {
                            visited[ni] = true;
                            bfs_queue.push({nx, ny, nz});
                        }
                    }
                }

                if (static_cast<int>(component.size()) < min_size) {
                    for (size_t ci : component)
                        labels[ci] = 0;
                }
            }
        }
    }
}

std::vector<int> Postprocessor::run(const std::vector<std::vector<float>>& patch_logits,
                                     const PatchGrid& grid)
{
    auto logits = aggregate_patches(patch_logits, grid);

    int C = grid.num_channels;
    int D = grid.vol_d, H = grid.vol_h, W = grid.vol_w;

    softmax_channels(logits, C, D, H, W);
    auto labels = argmax(logits, C, D, H, W);
    filter_small_components(labels, D, H, W, min_component_sz);

    return labels;
}
