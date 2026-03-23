#include "NiftiIO.h"
#include "Preprocessor.h"
#include "InferenceEngine.h"
#include "Postprocessor.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <array>
#include <algorithm>

namespace fs = std::filesystem;

struct Config {
    std::string input_dir;
    std::string output_path;
    std::string model_path;
    std::string device = "cpu";
    float patch_overlap = 0.5f;
    int min_component_size = 100;
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --input-dir <path> --output <path> --model <path>\n"
              << "  [--device cpu|cuda] [--patch-overlap 0.5] [--min-component-size 100]\n";
}

static Config parse_args(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--input-dir")             cfg.input_dir = next();
        else if (arg == "--output")           cfg.output_path = next();
        else if (arg == "--model")            cfg.model_path = next();
        else if (arg == "--device")           cfg.device = next();
        else if (arg == "--patch-overlap")    cfg.patch_overlap = std::stof(next());
        else if (arg == "--min-component-size") cfg.min_component_size = std::stoi(next());
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (cfg.input_dir.empty() || cfg.output_path.empty() || cfg.model_path.empty()) {
        std::cerr << "Error: --input-dir, --output, and --model are required.\n";
        print_usage(argv[0]);
        std::exit(1);
    }

    return cfg;
}

// Find _flair, _t1, _t1ce, _t2 nifti files in dir.
// _t1 match skips anything that already matched _t1ce.
static std::array<std::string, 4> find_modality_files(const std::string& dir) {
    std::array<std::string, 4> paths; // [flair, t1, t1ce, t2]

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();

        std::string lower = fname;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower.find(".nii") == std::string::npos) continue;

        if (lower.find("_flair") != std::string::npos) {
            paths[0] = entry.path().string();
        } else if (lower.find("_t1ce") != std::string::npos) {
            paths[2] = entry.path().string();
        } else if (lower.find("_t2") != std::string::npos) {
            paths[3] = entry.path().string();
        } else if (lower.find("_t1") != std::string::npos) {
            paths[1] = entry.path().string();
        }
    }

    const char* names[] = {"FLAIR", "T1", "T1ce", "T2"};
    for (size_t i = 0; i < 4; i++) {
        if (paths[i].empty()) {
            throw std::runtime_error(std::string("Could not find ") + names[i]
                + " modality in " + dir);
        }
    }

    return paths;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    auto wall_start = std::chrono::steady_clock::now();

    std::cerr << "Scanning " << cfg.input_dir << " for modality files...\n";
    auto mod_paths = find_modality_files(cfg.input_dir);

    const char* mod_names[] = {"FLAIR", "T1", "T1ce", "T2"};
    std::array<NiftiVolume, 4> volumes;
    for (size_t i = 0; i < 4; i++) {
        std::cerr << "  Loading " << mod_names[i] << ": " << mod_paths[i] << "\n";
        volumes[i] = nifti::load(mod_paths[i]);
    }

    std::cerr << "Volume dimensions: " << volumes[0].nx << " x " << volumes[0].ny
              << " x " << volumes[0].nz << "\n";

    Preprocessor preproc(128, cfg.patch_overlap);
    auto grid = preproc.run(volumes);
    std::cerr << "Extracted " << grid.patches.size() << " patches ("
              << grid.patch_d << "^3, overlap=" << cfg.patch_overlap << ")\n";

    InferenceEngine engine(cfg.model_path, cfg.device);
    std::vector<std::vector<float>> all_logits;
    all_logits.reserve(grid.patches.size());

    for (size_t i = 0; i < grid.patches.size(); i++) {
        std::cerr << "\rInference: patch " << (i + 1) << "/" << grid.patches.size() << std::flush;
        auto logits = engine.predict(grid.patches[i].data,
                                     grid.num_channels,
                                     grid.patch_d, grid.patch_h, grid.patch_w);
        all_logits.push_back(std::move(logits));
    }
    std::cerr << "\n";

    Postprocessor postproc(cfg.min_component_size);
    auto labels = postproc.run(all_logits, grid);

    nifti::save_labels(cfg.output_path, labels, volumes[0]);

    auto wall_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(wall_end - wall_start).count();
    std::cerr << "Saved segmentation to " << cfg.output_path << "\n";
    std::cerr << "Total time: " << elapsed << "s\n";

    return 0;
}
