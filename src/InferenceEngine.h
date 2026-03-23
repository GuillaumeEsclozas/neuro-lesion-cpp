#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

// ONNX Runtime wrapper. Takes [C,D,H,W] patches, returns raw logits.

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path, const std::string& device);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;

    /*
     * Run inference on a single patch. Input is a flat [C, D, H, W] buffer.
     * Returns raw logits as [num_classes, D, H, W].
     */
    std::vector<float> predict(const std::vector<float>& input_patch,
                               int channels, int depth, int height, int width);

    int output_channels() const { return out_channels; }

private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions session_opts;

    std::string input_name;
    std::string output_name;
    int out_channels = 4; /* BraTS: bg, NCR/NET, edema, ET */

    void query_io_names();
};
