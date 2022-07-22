// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "engine.h"

#include "device_agent.h"

//namespace nx {
//namespace vms_server_plugins {
//namespace analytics {
//namespace sample {
namespace sample {
    namespace analytics {
        namespace opencv_object_detection {

            using namespace nx::sdk;
            using namespace nx::sdk::analytics;

            Engine::Engine(std::filesystem::path pluginHomeDir):
                // Call the DeviceAgent helper class constructor telling it to verbosely report to stderr.
                nx::sdk::analytics::Engine(/*enableOutput*/ true),
                m_pluginHomeDir(pluginHomeDir),
                m_modelPath(std::move(pluginHomeDir))
            {
                //Engine::loadOnnxRuntimeModel();
                Engine::loadOnnxRuntimeModel();
            }

            Engine::~Engine()
            {
                Engine::releaseGPU();
            }

            /**
             * Called when the Server opens a video-connection to the camera if the plugin is enabled for this
             * camera.
             *
             * @param outResult The pointer to the structure which needs to be filled with the resulting value
             *     or the error information.
             * @param deviceInfo Contains various information about the related device such as its id, vendor,
             *     model, etc.
             */
            void Engine::doObtainDeviceAgent(Result<IDeviceAgent*>* outResult, const IDeviceInfo* deviceInfo)
            {
                //*outResult = new DeviceAgent(deviceInfo, m_pluginHomeDir);

                *outResult = new DeviceAgent(deviceInfo, m_pluginHomeDir, core_net);

            }

            /**
             * @return JSON with the particular structure. Note that it is possible to fill in the values
             *     that are not known at compile time, but should not depend on the Engine settings.
             */
            std::string Engine::manifestString() const
            {
                // Ask the Server to supply uncompressed video frames in YUV420 format (see
                // https://en.wikipedia.org/wiki/YUV).
                //
                // Note that this format is used internally by the Server, therefore requires minimum
                // resources for decoding, thus it is the recommended format.
                return /*suppress newline*/ 1 + (const char*) R"json(
            {
                "capabilities": "needUncompressedVideoFrames_bgr|deviceDependent",
                "preferredStream": "secondary"
            }
            )json";

            }

            // "capabilities": "needUncompressedVideoFrames_bgr|deviceDependent"
            //"capabilities": "needUncompressedVideoFrames_bgr",
            //"preferredStream" : "secondary"
            //"streamTypeFilter": "compressedVideo",

            void Engine::loadOnnxRuntimeModel()
            {
                static const auto modelBin = m_modelPath /
                    std::filesystem::path("yolov4_1_3_512_512_static_vfd_no_nms50.onnx");

                //static const auto modelBin = m_modelPath / std::filesystem::path("yolov4_1_3_416_416_static_dau_wt_nms.onnx");

                std::string str = modelBin.string();
                //std::string str = "C:/Program Files/DAEKYO CNS/VIVEex/MediaServer/plugins/opencv_object_detection_analytics_plugin/yolov4_1_3_416_416_static_50_items.ONNX";

                std::wstring wide_string = std::wstring(str.begin(), str.end());



                std::filesystem::create_directories( m_modelPath / "tensortRT_cache");
                static const auto tensorRT_path = m_modelPath / std::filesystem::path("tensortRT_cache");
                std::basic_string c_path = tensorRT_path.string();

                //const char* cache_path = tensorRT_path.string().c_str();
                


                if (with_cuda)
                {
                    cuda_options.device_id = 0;
                    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                    cuda_options.gpu_mem_limit = static_cast<int>(1 * 1024 * 1024 * 1024);  // 1 Gb
                    cuda_options.arena_extend_strategy = 1;
                    cuda_options.do_copy_in_default_stream = 1;
                    cuda_options.has_user_compute_stream = 1;
                    cuda_options.default_memory_arena_cfg = nullptr;

                    trt_options.device_id = 0;
                    //trt_options.trt_max_workspace_size = 0.5 * 1024 * 1024 * 1024;
                    //trt_options.trt_max_partition_iterations = 10;
                    //trt_options.trt_min_subgraph_size = 5;
                    //trt_options.trt_fp16_enable = 1;
                    //trt_options.trt_int8_enable = 1;
                    trt_options.trt_int8_use_native_calibration_table = 1;
                    trt_options.trt_engine_cache_enable = 1;
                    const char* cache_path = c_path.c_str();
                    trt_options.trt_engine_cache_path = cache_path;
                    trt_options.trt_dump_subgraphs = 1;

                    //sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

                    sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
                    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
                    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
                    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
                    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                }

                sessionOptions.SetIntraOpNumThreads(4);
                sessionOptions.SetInterOpNumThreads(4);

                std::basic_string<ORTCHAR_T> onnxpath = std::basic_string<ORTCHAR_T>(wide_string);
                core_net = new Ort::Session(env, onnxpath.c_str(), sessionOptions);
            }

            void Engine::releaseGPU()
            {
                core_net->release();
            }

        } // namespace opencv_object_detection
    } // namespace analytics
} // namespace sample

//} // namespace sample
//} // namespace analytics
//} // namespace vms_server_plugins
//} // namespace nx
