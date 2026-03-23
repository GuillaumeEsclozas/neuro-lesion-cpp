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
    int C = grid.num_channels;
    int pD = grid.padded_d, pH = grid.padded_h, pW = grid.padded_w;
    size_t vol_total = (size_t)C * pD * pH * pW;

    std::vector<float> accum(vol_total, 0.0f);
    std::vector<float> counts((size_t)pD * pH * pW, 0.0f);

    int psz = grid.patch_d;

    for (size_t pi = 0; pi < grid.patches.size(); pi++) {
        const Patch& patch = grid.patches[pi];
        const auto& logits = patch_logits[pi];
        int ox = patch.origin_x, oy = patch.origin_y, oz = patch.origin_z;

        for (int c = 0; c < C; c++) {
            for (int dz = 0; dz < psz; dz++) {
                for (int dy = 0; dy < psz; dy++) {
                    for (int dx = 0; dx < psz; dx++) {
                        size_t src_idx = (size_t)c * psz * psz * psz
                                       + (size_t)dz * psz * psz
                                       + (size_t)dy * psz + dx;

                        int gz = oz + dz, gy = oy + dy, gx = ox + dx;
                        size_t dst_idx = (size_t)c * pD * pH * pW
                                       + (size_t)gz * pH * pW
                                       + (size_t)gy * pW + gx;

                        accum[dst_idx] += logits[src_idx];

                        if (c == 0) {
                            size_t cnt_idx = (size_t)gz * pH * pW + (size_t)gy * pW + gx;
                            counts[cnt_idx] += 1.0f;
                        }
                    }
                }
            }
        }
    }

    // average over overlap count
    for (int c = 0; c < C; c++) {
        for (size_t i = 0; i < (size_t)pD * pH * pW; i++) {
            if (counts[i] > 0.0f) {
                accum[(size_t)c * pD * pH * pW + i] /= counts[i];
            }
        }
    }

    // crop back to original dims
    int D = grid.vol_d, H = grid.vol_h, W = grid.vol_w;
    std::vector<float> result((size_t)C * D * H * W);

    for (int c = 0; c < C; c++) {
        for (int z = 0; z < D; z++) {
            for (int y = 0; y < H; y++) {
                size_t src_off = (size_t)c * pD * pH * pW + (size_t)z * pH * pW + (size_t)y * pW;
                size_t dst_off = (size_t)c * D * H * W + (size_t)z * H * W + (size_t)y * W;
                std::copy(&accum[src_off], &accum[src_off + W], &result[dst_off]);
            }
        }
    }

    return result;
}

void Postprocessor::softmax_channels(std::vector<float>& probs, int C, int D, int H, int W) {
    size_t spatial = (size_t)D * H * W;

    for (size_t i = 0; i < spatial; i++) {
        float max_val = probs[i];
        for (int c = 1; c < C; c++) {
            float v = probs[c * spatial + i];
            if (v > max_val) max_val = v;
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C; c++) {
            float e = std::exp(probs[c * spatial + i] - max_val);
            probs[c * spatial + i] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < C; c++) {
            probs[c * spatial + i] *= inv_sum;
        }
    }
}

std::vector<int> Postprocessor::argmax(const std::vector<float>& probs, int C, int D, int H, int W) {
    size_t spatial = (size_t)D * H * W;
    std::vector<int> labels(spatial);

    for (size_t i = 0; i < spatial; i++) {
        int best = 0;
        float best_val = probs[i];
        for (int c = 1; c < C; c++) {
            float v = probs[c * spatial + i];
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
    size_t total = (size_t)D * H * W;
    std::vector<bool> visited(total, false);

    const int dx[] = {-1, 1,  0, 0,  0, 0};
    const int dy[] = { 0, 0, -1, 1,  0, 0};
    const int dz[] = { 0, 0,  0, 0, -1, 1};

    auto idx = [&](int x, int y, int z) -> size_t {
        return (size_t)z * H * W + (size_t)y * W + x;
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
                        int nx = cx + dx[d], ny = cy + dy[d], nz = cz + dz[d];
                        if (nx < 0 || nx >= W || ny < 0 || ny >= H || nz < 0 || nz >= D)
                            continue;
                        size_t ni = idx(nx, ny, nz);
                        if (!visited[ni] && labels[ni] == lbl) {
                            visited[ni] = true;
                            bfs_queue.push({nx, ny, nz});
                        }
                    }
                }

                // kill small blobs
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
