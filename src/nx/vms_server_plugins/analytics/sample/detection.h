#pragma once
#include <nx/sdk/analytics/rect.h>
#include <nx/sdk/uuid.h>
#include <string>
#include <memory>
#include <vector>

namespace sample {
    namespace analytics {
        namespace opencv_object_detection {

            // Class labels for the MobileNet SSD model (VOC dataset).
            extern const std::vector<std::string> kClasses;
            extern const std::vector<std::string> kClassesToDetect;

            struct Detection
            {
                nx::sdk::analytics::Rect boundingBox;
                std::string classLabel;
                float confidence;
                nx::sdk::Uuid trackId;
            };

            using DetectionList = std::vector<std::shared_ptr<Detection>>;

        } // namespace opencv_object_detection
    } // namespace analytics
} // namespace sample