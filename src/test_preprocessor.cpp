#include "Preprocessor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

#define TEST(name) static void name(); \
    static struct name##_reg { name##_reg() { tests.push_back({#name, name}); } } name##_inst; \
    static void name()

struct TestEntry { const char* name; void(*fn)(); };
static std::vector<TestEntry> tests;

static bool approx(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

// ── z-score normalization ──────────────────────────────────────────────

TEST(zscore_basic) {
    NiftiVolume vol;
    vol.nx = 4; vol.ny = 4; vol.nz = 1;
    vol.dx = vol.dy = vol.dz = 1.0f;
    vol.data.resize(16);

    // Set half to zero (background), half to known values
    for (int i = 0; i < 8; i++) vol.data[i] = 0.0f;
    vol.data[8]  = 10.0f;
    vol.data[9]  = 20.0f;
    vol.data[10] = 30.0f;
    vol.data[11] = 40.0f;
    vol.data[12] = 50.0f;
    vol.data[13] = 60.0f;
    vol.data[14] = 70.0f;
    vol.data[15] = 80.0f;

    Preprocessor::zscore_normalize(vol);

    // Background voxels must remain zero
    for (int i = 0; i < 8; i++) {
        assert(vol.data[i] == 0.0f);
    }

    // Nonzero voxels: mean=45, std~22.36
    // After normalization, mean of nonzero should be ~0
    double sum = 0;
    int cnt = 0;
    for (int i = 8; i < 16; i++) {
        sum += vol.data[i];
        cnt++;
    }
    assert(approx(static_cast<float>(sum / cnt), 0.0f, 0.01f));

    // Std should be ~1
    double sq_sum = 0;
    for (int i = 8; i < 16; i++) sq_sum += vol.data[i] * vol.data[i];
    float std_val = std::sqrt(static_cast<float>(sq_sum / cnt));
    assert(approx(std_val, 1.0f, 0.01f));
}

TEST(zscore_all_zero) {
    NiftiVolume vol;
    vol.nx = 2; vol.ny = 2; vol.nz = 2;
    vol.dx = vol.dy = vol.dz = 1.0f;
    vol.data.assign(8, 0.0f);

    Preprocessor::zscore_normalize(vol);

    // Nothing should change
    for (auto v : vol.data) assert(v == 0.0f);
}

// ── patch extraction ────────────────────────────────────────────────────

TEST(single_patch_small_volume) {
    // Volume smaller than patch size should produce exactly one patch
    int C = 2, D = 4, H = 4, W = 4;
    std::vector<float> data(C * D * H * W);
    std::iota(data.begin(), data.end(), 0.0f);

    Preprocessor p(8, 0.5f); // patch_size=8, overlap doesn't matter here
    auto grid = p.extract_patches(data, C, D, H, W);

    assert(grid.patches.size() == 1);
    assert(grid.patch_d == 8);
    assert(grid.padded_d >= D);
    assert(grid.padded_h >= H);
    assert(grid.padded_w >= W);

    // The original data should appear in the top-left corner of the patch
    auto& patch = grid.patches[0];
    // Check first element of each channel
    for (int c = 0; c < C; c++) {
        float expected = static_cast<float>(c * D * H * W);
        assert(patch.data[c * 8 * 8 * 8] == expected);
    }
}

TEST(overlap_produces_multiple_patches) {
    int C = 1, D = 12, H = 12, W = 12;
    std::vector<float> data(C * D * H * W, 1.0f);

    Preprocessor p(8, 0.5f); // step = 4
    auto grid = p.extract_patches(data, C, D, H, W);

    // With D=12, patch=8, step=4: positions at z=0,4,8 but 8+8=16 > padded_d
    // padded_d should be 12 -> positions 0,4. That gives 2 per dim.
    // Actually let's just check we get more than 1
    assert(grid.patches.size() > 1);

    // All patches from a constant volume should have all 1s in the valid region
    for (auto& patch : grid.patches) {
        // First element should be 1.0 if it falls within original volume
        if (patch.origin_x < W && patch.origin_y < H && patch.origin_z < D) {
            assert(patch.data[0] == 1.0f);
        }
    }
}

TEST(zscore_single_nonzero_voxel) {
    // Only one nonzero voxel: count < 2, normalization should be skipped
    NiftiVolume vol;
    vol.nx = 2; vol.ny = 2; vol.nz = 1;
    vol.data.assign(4, 0.0f);
    vol.data[0] = 42.0f;

    Preprocessor::zscore_normalize(vol);
    assert(vol.data[0] == 42.0f); // unchanged
    assert(vol.data[1] == 0.0f);
}

TEST(patch_boundary_non_divisible) {
    // Volume dimensions not divisible by patch size or step.
    // 10x10x10 with patch_size=8, overlap=0.5 (step=4).
    // pad_dim(10): n_patches=(10-8+4-1)/4+1 = 5/4+1 = 1+1=2, padded=(2-1)*4+8=12
    int C = 1, D = 10, H = 10, W = 10;
    std::vector<float> data(C * D * H * W, 1.0f);
    // Set a known voxel at the far corner
    data[9 * H * W + 9 * W + 9] = 7.0f; // (9,9,9)

    Preprocessor p(8, 0.5f);
    auto grid = p.extract_patches(data, C, D, H, W);

    assert(grid.padded_d == 12);
    assert(grid.padded_h == 12);
    assert(grid.padded_w == 12);
    // 2 positions per dim: 0,4. Total 2^3=8 patches
    assert(grid.patches.size() == 8);

    // The patch starting at (4,4,4) should contain the (9,9,9) voxel
    // at local offset (5,5,5) in that patch
    bool found = false;
    for (auto& patch : grid.patches) {
        if (patch.origin_x == 4 && patch.origin_y == 4 && patch.origin_z == 4) {
            // local (5,5,5): idx = 0*8*8*8 + 5*8*8 + 5*8 + 5 = 365
            assert(approx(patch.data[5 * 8 * 8 + 5 * 8 + 5], 7.0f));
            found = true;
        }
    }
    assert(found);

    // Padded region (beyond original volume) should be zero.
    // Patch at (4,4,4): local (7,7,7) maps to global (11,11,11) which is padding.
    for (auto& patch : grid.patches) {
        if (patch.origin_x == 4 && patch.origin_y == 4 && patch.origin_z == 4) {
            assert(patch.data[7 * 8 * 8 + 7 * 8 + 7] == 0.0f);
        }
    }
}

TEST(exact_patch_size_volume) {
    // Volume exactly matches patch size: should produce exactly 1 patch with no padding
    int C = 1, D = 8, H = 8, W = 8;
    std::vector<float> data(C * D * H * W);
    std::iota(data.begin(), data.end(), 0.0f);

    Preprocessor p(8, 0.5f);
    auto grid = p.extract_patches(data, C, D, H, W);

    assert(grid.patches.size() == 1);
    assert(grid.padded_d == 8);
    assert(grid.padded_h == 8);
    assert(grid.padded_w == 8);

    // Data should be a verbatim copy
    assert(grid.patches[0].data.size() == data.size());
    for (size_t i = 0; i < data.size(); i++) {
        assert(grid.patches[0].data[i] == data[i]);
    }
}

TEST(stacking_preserves_channels) {
    std::array<NiftiVolume, 4> vols;
    for (int c = 0; c < 4; c++) {
        vols[c].nx = 2; vols[c].ny = 2; vols[c].nz = 2;
        vols[c].dx = vols[c].dy = vols[c].dz = 1.0f;
        vols[c].data.assign(8, static_cast<float>(c + 1) * 100.0f);
    }

    auto stacked = Preprocessor::stack_modalities(vols);
    assert(stacked.size() == 4 * 8);

    // Each channel should have its constant value
    for (int c = 0; c < 4; c++) {
        float expected = (c + 1) * 100.0f;
        for (int i = 0; i < 8; i++) {
            assert(stacked[c * 8 + i] == expected);
        }
    }
}

TEST(stacking_rejects_mismatched_dims) {
    std::array<NiftiVolume, 4> vols;
    for (int c = 0; c < 4; c++) {
        vols[c].nx = 4; vols[c].ny = 4; vols[c].nz = 4;
        vols[c].data.assign(64, 1.0f);
    }
    vols[2].nx = 5; // break channel 2
    vols[2].data.assign(5 * 4 * 4, 1.0f);

    bool threw = false;
    try {
        Preprocessor::stack_modalities(vols);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    int passed = 0, failed = 0;
    for (auto& t : tests) {
        try {
            t.fn();
            std::cout << "  PASS  " << t.name << "\n";
            passed++;
        } catch (const std::exception& e) {
            std::cout << "  FAIL  " << t.name << ": " << e.what() << "\n";
            failed++;
        } catch (...) {
            std::cout << "  FAIL  " << t.name << " (unknown exception)\n";
            failed++;
        }
    }
    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed > 0 ? 1 : 0;
}
