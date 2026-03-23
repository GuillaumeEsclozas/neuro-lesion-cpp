#include "InferenceEngine.h"
#include <stdexcept>
#include <iostream>

InferenceEngine::InferenceEngine(const std::string& model_path, const std::string& device)
    : env(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "lesion_seg"))
{
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_opts.SetIntraOpNumThreads(0); // let ORT pick

    if (device == "cuda") {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        cuda_opts.arena_extend_strategy = 0;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        cuda_opts.do_copy_in_default_stream = 1;
        session_opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cerr << "[InferenceEngine] Using CUDA execution provider\n";
    } else {
        session_opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        std::cerr << "[InferenceEngine] Using CPU execution provider\n";
    }

    // ORT on Windows needs wide strings, on Linux narrow is fine
#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    session = std::make_unique<Ort::Session>(*env, wpath.c_str(), session_opts);
#else
    session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_opts);
#endif

    query_io_names();
}

InferenceEngine::~InferenceEngine() = default;
InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;
InferenceEngine& InferenceEngine::operator=(InferenceEngine&&) noexcept = default;

void InferenceEngine::query_io_names() {
    Ort::AllocatorWithDefaultOptions alloc;

    auto in_name = session->GetInputNameAllocated(0, alloc);
    input_name = in_name.get();

    auto out_name = session->GetOutputNameAllocated(0, alloc);
    output_name = out_name.get();

    // Try to read the number of output channels from the model metadata
    auto out_info = session->GetOutputTypeInfo(0);
    auto tensor_info = out_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    // Expected shape: [batch, num_classes, D, H, W]
    if (shape.size() == 5 && shape[1] > 0) {
        out_channels = static_cast<int>(shape[1]);
    }
}

std::vector<float> InferenceEngine::predict(const std::vector<float>& input_patch,
                                             int channels, int depth, int height, int width)
{
    std::array<int64_t, 5> input_shape = {1, channels, depth, height, width};
    size_t input_count = (size_t)channels * depth * height * width;

    if (input_patch.size() != input_count)
        throw std::runtime_error("Input patch size mismatch: expected "
            + std::to_string(input_count) + ", got " + std::to_string(input_patch.size()));

    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(input_patch.data()),
        input_count,
        input_shape.data(),
        input_shape.size()
    );

    const char* in_names[]  = { input_name.c_str() };
    const char* out_names[] = { output_name.c_str() };

    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        in_names, &input_tensor, 1,
        out_names, 1
    );

    float* raw = output_tensors[0].GetTensorMutableData<float>();
    auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t total = 1;
    for (auto s : out_shape) total *= s;

    return std::vector<float>(raw, raw + total);
}
