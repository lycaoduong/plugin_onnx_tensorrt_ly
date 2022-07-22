// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <nx/sdk/analytics/helpers/plugin.h>
#include <nx/sdk/analytics/helpers/engine.h>
#include <nx/sdk/analytics/i_uncompressed_video_frame.h>
#include <filesystem>

//namespace nx {
//namespace vms_server_plugins {
//namespace analytics {
//namespace sample {

//#include "object_detector.h"
//#include "object_tracker.h"
#include "onnxruntime_cxx_api.h"
#include "provider_options.h"
#include "tensorrt_provider_factory.h"


namespace sample {
    namespace analytics {
        namespace opencv_object_detection {

            class Engine: public nx::sdk::analytics::Engine
            {
            public:
                explicit Engine(std::filesystem::path pluginHomeDir);
                //Engine();
                virtual ~Engine() override;

            protected:
                virtual std::string manifestString() const override;

            protected:
                virtual void doObtainDeviceAgent(
                    nx::sdk::Result<nx::sdk::analytics::IDeviceAgent*>* outResult,
                    const nx::sdk::IDeviceInfo* deviceInfo) override;

            private:
                std::filesystem::path m_pluginHomeDir;
                // Ly
                Ort::Session* core_net;
                //Ort::Session* core_net0;
                //Ort::Session* core_net1;
                //Ort::Session* core_net2;
                //Ort::Session* core_net3;
                //Ort::Session* core_net4;

                Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "TextModel");
                bool with_cuda = true;
                Ort::SessionOptions sessionOptions;
                OrtCUDAProviderOptions cuda_options;
                OrtTensorRTProviderOptions trt_options{};
                const std::filesystem::path m_modelPath;
                void loadOnnxRuntimeModel();


                void releaseGPU();
                //static constexpr int kModel = 5;
                //int m_coreIndex = 0;
            };

        } // namespace opencv_object_detection
    } // namespace analytics
} // namespace sample

//} // namespace sample
//} // namespace analytics
//} // namespace vms_server_plugins
//} // namespace nx
