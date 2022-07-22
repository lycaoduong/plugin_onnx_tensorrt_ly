#pragma once


#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>

#include <nx/sdk/helpers/uuid_helper.h>
#include <nx/sdk/uuid.h>

#include "detection.h"
#include "frame.h"

#include "onnxruntime_cxx_api.h"

namespace sample {
    namespace analytics {
        namespace opencv_object_detection {
            class ObjectDetector
            {
            public:
                //explicit ObjectDetector(std::filesystem::path modelPath);
                explicit ObjectDetector(Ort::Session* core_net);
                //explicit ObjectDetector();

                void ensureInitialized();
                bool isTerminated() const;
                void terminate();
                DetectionList run(const Frame& frame);
                //int vectorProduct(std::vector<int64_t> V);

            private:
                //void loadModel();
                void loadOnnxRuntimeModel();
                //DetectionList runImpl(const Frame& frame);
                //DetectionList runImplCustom(const Frame& frame);
                //DetectionList runYoloOnnx(const Frame& frame);
                DetectionList runYoloOnnx_custom(const Frame& frame);

            private:
                bool m_netLoaded = false;
                bool m_terminated = false;
                const std::filesystem::path m_modelPath;
                //std::unique_ptr<cv::dnn::Net> m_net;
                Ort::Session* onnx_net;
                Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

                float confidence_threshold = 0.3;
                float human_threshold = 0.7;
                float nmsThreshold = 0.4;

                //Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "TextModel");
                //Ort::SessionOptions sessionOptions;
                //Ort::Session* onnx_net;
                //Ort::AllocatorWithDefaultOptions allocator;
                //OrtCUDAProviderOptions cuda_options;
                //Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                //    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

                // Onnx Yolo
                int img_size = 512;
                int out_shape = 16128;
                cv::Mat blob;
                cv::Size yoloInputImageSize = cv::Size(img_size, img_size);

                std::vector<int64_t> inputDims{ 1, 3, img_size, img_size };
                size_t inputTensorSize = 3 * img_size * img_size;
                std::vector<float> inputTensorValues = std::vector<float>(inputTensorSize);
                std::vector<Ort::Value> inputTensors;

                std::vector<int64_t> outputDims{ out_shape, 6 };
                size_t outputTensorSize = out_shape * 6;
                std::vector<float> outputTensorValues = std::vector<float>(outputTensorSize);
                std::vector<Ort::Value> outputTensors;

                //std::vector<int64_t> outputBBDims{ 1, 10647, 1, 4 };
                //size_t outputBBTensorSize = 42588;
                //std::vector<float> outputBBTensorValues = std::vector<float>(42588);
                //std::vector<Ort::Value> outputBBTensors;

                //std::vector<int64_t> outputCfDims{ 1, 10647, 3 };
                //size_t outputCfTensorSize = 31941;
                //std::vector<float> outputCfTensorValues = std::vector<float>(31941);
                //std::vector<Ort::Value> outputCfTensors;

                const char* inputName = "input";
                const char* outputName = "output";
                //const char* outputBBName = "boxes";
                //const char* outputCfName = "confs";
                bool with_cuda = true;

                // Onnx EfficientDt
                //cv::Size netInputImageSize = cv::Size(512, 512);
                //double scaleFactor = 1.0 / 255.0;
                //std::vector<int64_t> inputDims{ 1, 3, 512, 512 };
                //std::vector<int64_t> outputDims{ 20, 6 };
                //size_t inputTensorSize = 786432;
                //size_t outputTensorSize = 120;
                //std::vector<float> inputTensorValues = std::vector<float> (786432);
                //std::vector<float> outputTensorValues = std::vector<float>(120);
                //std::vector<Ort::Value> inputTensors;
                //std::vector<Ort::Value> outputTensors;

                nx::sdk::Uuid m_trackId = nx::sdk::UuidHelper::randomUuid();
            };
        } // namespace opencv_object_detection
    } // namespace analytics
} // namespace sample

