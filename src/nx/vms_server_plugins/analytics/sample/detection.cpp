#include "detection.h"

namespace sample {
    namespace analytics {
        namespace opencv_object_detection {

                // Class labels for the MobileNet SSD model (VOC dataset).
                //const std::vector<std::string> kClasses{
                //    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                //    "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant",
                //    "sheep", "sofa", "train", "tv monitor"
                //};
                //const std::vector<std::string> kClassesToDetect{"cat", "dog", "person" };

                const std::vector<std::string> kClasses{ 
                    "human", "helmet", "safety vest", "face shield", "safety shoes", "safety uniform", 
                    "safety boots", "gloves", "overall" , "smoke", "head", "fire" ,
                    "human_fall", "coat", "apron", "non_helmet", "half-fall"
                };
                //const std::vector<std::string> kClasses{
                //"fire", "smoke", "human"
                //};
                const std::vector<std::string> kClassesToDetect{
                    "helmet", "smoke", "fire", "head", "human_fall", 
                    "non_helmet", "human", "half-fall" };
                //const std::vector<std::string> kClassesToDetect{
                //"fire", "smoke", "human"};

        } // namespace opencv_object_detection
    } // namespace analytics
} // namespace sample