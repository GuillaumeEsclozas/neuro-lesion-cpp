#include "Postprocessor.h"
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

TEST(softmax_uniform) {
    int C = 4, D = 2, H = 2, W = 2;
    size_t spatial = static_cast<size_t>(D) * static_cast<size_t>(H) * static_cast<size_t>(W);
    std::vector<float> probs(static_cast<size_t>(C) * spatial, 1.0f);

    Postprocessor::softmax_channels(probs, C, D, H, W);

    for (size_t i = 0; i < spatial; i++) {
        float sum = 0;
        for (size_t c = 0; c < static_cast<size_t>(C); c++) {
            assert(approx(probs[c * spatial + i], 0.25f));
            sum += probs[c * spatial + i];
        }
        assert(approx(sum, 1.0f));
    }
}

TEST(softmax_dominant_channel) {
    int C = 3, D = 1, H = 1, W = 1;
    std::vector<float> probs = {10.0f, 0.0f, 0.0f};

    Postprocessor::softmax_channels(probs, C, D, H, W);

    assert(probs[0] > 0.99f);
    assert(probs[1] < 0.01f);
    assert(probs[2] < 0.01f);
    assert(approx(probs[0] + probs[1] + probs[2], 1.0f, 0.001f));
}

TEST(argmax_basic) {
    int C = 4, D = 1, H = 2, W = 2;
    size_t spatial = 4;
    std::vector<float> probs(static_cast<size_t>(C) * spatial, 0.0f);

    probs[0 * spatial + 0] = 0.9f;
    probs[1 * spatial + 1] = 0.9f;
    probs[2 * spatial + 2] = 0.9f;
    probs[3 * spatial + 3] = 0.9f;

    auto labels = Postprocessor::argmax(probs, C, D, H, W);

    assert(labels[0] == 0);
    assert(labels[1] == 1);
    assert(labels[2] == 2);
    assert(labels[3] == 3);
}

TEST(filter_removes_tiny_blobs) {
    int D = 1, H = 4, W = 4;
    std::vector<int> labels(16, 0);

    labels[0] = 1;
    labels[1] = 1;

    labels[8]  = 2;
    labels[9]  = 2;
    labels[12] = 2;
    labels[13] = 2;

    Postprocessor::filter_small_components(labels, D, H, W, 3);

    assert(labels[0] == 0);
    assert(labels[1] == 0);

    assert(labels[8]  == 2);
    assert(labels[9]  == 2);
    assert(labels[12] == 2);
    assert(labels[13] == 2);
}

TEST(filter_keeps_large_components) {
    int D = 1, H = 1, W = 10;
    std::vector<int> labels(10, 1);

    Postprocessor::filter_small_components(labels, D, H, W, 5);

    for (size_t i = 0; i < 10; i++)
        assert(labels[i] == 1);
}

TEST(filter_3d_connectivity) {
    int D = 2, H = 2, W = 2;
    std::vector<int> labels(8, 0);
    labels[0] = 1;
    labels[4] = 1;

    Postprocessor::filter_small_components(labels, D, H, W, 2);

    assert(labels[0] == 1);
    assert(labels[4] == 1);

    Postprocessor::filter_small_components(labels, D, H, W, 3);

    assert(labels[0] == 0);
    assert(labels[4] == 0);
}

TEST(aggregate_single_patch) {
    PatchGrid grid;
    grid.patch_d = grid.patch_h = grid.patch_w = 2;
    grid.num_channels = 2;
    grid.vol_d = grid.vol_h = grid.vol_w = 2;
    grid.padded_d = grid.padded_h = grid.padded_w = 2;

    Patch p;
    p.origin_x = p.origin_y = p.origin_z = 0;
    p.data.resize(2 * 2 * 2 * 2);
    std::iota(p.data.begin(), p.data.end(), 1.0f);
    grid.patches.push_back(p);

    std::vector<std::vector<float>> logits = {p.data};
    auto result = Postprocessor::aggregate_patches(logits, grid);

    assert(result.size() == 2 * 2 * 2 * 2);
    for (size_t i = 0; i < result.size(); i++) {
        assert(approx(result[i], static_cast<float>(i + 1)));
    }
}

TEST(filter_single_voxel_component) {
    int D = 1, H = 3, W = 3;
    std::vector<int> labels(9, 0);
    labels[4] = 1;

    Postprocessor::filter_small_components(labels, D, H, W, 2);
    assert(labels[4] == 0);

    labels[4] = 1;
    Postprocessor::filter_small_components(labels, D, H, W, 1);
    assert(labels[4] == 1);
}

TEST(argmax_single_voxel) {
    int C = 4, D = 1, H = 1, W = 1;
    std::vector<float> probs = {0.1f, 0.5f, 0.3f, 0.1f};
    auto labels = Postprocessor::argmax(probs, C, D, H, W);
    assert(labels.size() == 1);
    assert(labels[0] == 1);
}

TEST(softmax_single_class) {
    int C = 1, D = 2, H = 2, W = 2;
    std::vector<float> probs(8, 42.0f);
    Postprocessor::softmax_channels(probs, C, D, H, W);
    for (size_t i = 0; i < probs.size(); i++) assert(approx(probs[i], 1.0f));
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
