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

TEST(zscore_basic) {
    NiftiVolume vol;
    vol.nx = 4; vol.ny = 4; vol.nz = 1;
    vol.dx = vol.dy = vol.dz = 1.0f;
    vol.data.resize(16);

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

    for (int i = 0; i < 8; i++) {
        assert(vol.data[i] == 0.0f);
    }

    double sum = 0;
    int cnt = 0;
    for (int i = 8; i < 16; i++) {
        sum += vol.data[i];
        cnt++;
    }
    assert(approx(static_cast<float>(sum / cnt), 0.0f, 0.01f));

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

    for (auto v : vol.data) assert(v == 0.0f);
}

TEST(single_patch_small_volume) {
    int C = 2, D = 4, H = 4, W = 4;
    std::vector<float> data(C * D * H * W);
    std::iota(data.begin(), data.end(), 0.0f);

    Preprocessor p(8, 0.5f);
    auto grid = p.extract_patches(data, C, D, H, W);

    assert(grid.patches.size() == 1);
    assert(grid.patch_d == 8);
    assert(grid.padded_d >= D);
    assert(grid.padded_h >= H);
    assert(grid.padded_w >= W);

    auto& patch = grid.patches[0];
    for (int c = 0; c < C; c++) {
        float expected = static_cast<float>(c * D * H * W);
        assert(patch.data[c * 8 * 8 * 8] == expected);
    }
}

TEST(overlap_produces_multiple_patches) {
    int C = 1, D = 12, H = 12, W = 12;
    std::vector<float> data(C * D * H * W, 1.0f);

    Preprocessor p(8, 0.5f);
    auto grid = p.extract_patches(data, C, D, H, W);

    assert(grid.patches.size() > 1);

    for (auto& patch : grid.patches) {
        if (patch.origin_x < W && patch.origin_y < H && patch.origin_z < D) {
            assert(patch.data[0] == 1.0f);
        }
    }
}

TEST(zscore_single_nonzero_voxel) {
    NiftiVolume vol;
    vol.nx = 2; vol.ny = 2; vol.nz = 1;
    vol.data.assign(4, 0.0f);
    vol.data[0] = 42.0f;

    Preprocessor::zscore_normalize(vol);
    assert(vol.data[0] == 42.0f);
    assert(vol.data[1] == 0.0f);
}

TEST(patch_boundary_non_divisible) {
    int C = 1, D = 10, H = 10, W = 10;
    std::vector<float> data(C * D * H * W, 1.0f);
    data[9 * H * W + 9 * W + 9] = 7.0f;

    Preprocessor p(8, 0.5f);
    auto grid = p.extract_patches(data, C, D, H, W);

    assert(grid.padded_d == 12);
    assert(grid.padded_h == 12);
    assert(grid.padded_w == 12);
    assert(grid.patches.size() == 8);

    bool found = false;
    for (auto& patch : grid.patches) {
        if (patch.origin_x == 4 && patch.origin_y == 4 && patch.origin_z == 4) {
            assert(approx(patch.data[5 * 8 * 8 + 5 * 8 + 5], 7.0f));
            found = true;
        }
    }
    assert(found);

    for (auto& patch : grid.patches) {
        if (patch.origin_x == 4 && patch.origin_y == 4 && patch.origin_z == 4) {
            assert(patch.data[7 * 8 * 8 + 7 * 8 + 7] == 0.0f);
        }
    }
}

TEST(exact_patch_size_volume) {
    int C = 1, D = 8, H = 8, W = 8;
    std::vector<float> data(C * D * H * W);
    std::iota(data.begin(), data.end(), 0.0f);

    Preprocessor p(8, 0.5f);
    auto grid = p.extract_patches(data, C, D, H, W);

    assert(grid.patches.size() == 1);
    assert(grid.padded_d == 8);
    assert(grid.padded_h == 8);
    assert(grid.padded_w == 8);

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

    for (int c = 0; c < 4; c++) {
        float expected = static_cast<float>(c + 1) * 100.0f;
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
    vols[2].nx = 5;
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
