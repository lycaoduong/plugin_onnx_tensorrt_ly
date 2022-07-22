
#include "object_detector.h"
#include <opencv2/core.hpp>
#include "exceptions.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

#include<string>
#include <vector>
#include<numeric>


namespace sample {
    namespace analytics {
        namespace opencv_object_detection {

            using namespace std::string_literals;

            using namespace cv;
            using namespace cv::dnn;


            //ObjectDetector::ObjectDetector(std::filesystem::path modelPath):
            //    m_modelPath(std::move(modelPath))
            //{
            //}

            ObjectDetector::ObjectDetector(Ort::Session* core_net):
                onnx_net(core_net)
            {
            }

            //int ObjectDetector::vectorProduct(std::vector<int64_t> V)
            //{
            //    return accumulate(V.begin(), V.end(), 1, std::multiplies<int>());
            //}

            int vectorProduct(std::vector<int64_t> V)
            {
                return accumulate(V.begin(), V.end(), 1, std::multiplies<int>());
            }

            /**
             * Load the model if it is not loaded, do nothing otherwise. In case of errors terminate the
             * plugin and throw a specialized exception.
             */
            void ObjectDetector::ensureInitialized()
            {
                if (isTerminated())
                {
                    throw ObjectDetectorIsTerminatedError(
                        "Object detector initialization error: object detector is terminated.");
                }
                if (m_netLoaded)
                    return;

                try
                {
                    //loadModel();
                    loadOnnxRuntimeModel();
                }
                catch (const cv::Exception& e)
                {
                    terminate();
                    throw ObjectDetectorInitializationError("Loading model: " + cvExceptionToStdString(e));
                }
                catch (const std::exception& e)
                {
                    terminate();
                    throw ObjectDetectorInitializationError("Loading model: Error: "s + e.what());
                }
            }


            bool ObjectDetector::isTerminated() const
            {
                return m_terminated;
            }

            void ObjectDetector::terminate()
            {
                m_terminated = true;
            }

            DetectionList ObjectDetector::run(const Frame& frame)
            {
                if (isTerminated())
                    throw ObjectDetectorIsTerminatedError("Detection error: object detector is terminated.");

                try
                {
                    //return runImpl(frame);
                    return runYoloOnnx_custom(frame);
                }
                catch (const cv::Exception& e)
                {
                    terminate();
                    throw ObjectDetectionError(cvExceptionToStdString(e));
                }
                catch (const std::exception& e)
                {
                    terminate();
                    throw ObjectDetectionError("Error: "s + e.what());
                }
            }

            //-------------------------------------------------------------------------------------------------
            // private

            //void ObjectDetector::loadModel()
            //{
            //    // Prepare paths of model weights and definition.
            //    static const auto modelBin = m_modelPath /
            //        std::filesystem::path("MobileNetSSD.caffemodel");
            //    static const auto modelTxt = m_modelPath /
            //        std::filesystem::path("MobileNetSSD.prototxt");

            //    // Load the model for future processing using OpenCV.
            //    m_net = std::make_unique<Net>(
            //        readNetFromCaffe(modelTxt.string(), modelBin.string()));

            //    // Save the whether the net is loaded or not to prevent unnecessary load.
            //    m_netLoaded = !m_net->empty();

            //    if (!m_netLoaded)
            //        throw ObjectDetectorInitializationError("Loading model: network is empty.");
            //}

            void ObjectDetector::loadOnnxRuntimeModel()
            {
                std::string str = "No need to load model here, check on Engine.c";

                //static const auto modelBin = m_modelPath /
                //    std::filesystem::path("yolov4_1_3_416_416_static_50_items.ONNX");

                //std::string str = modelBin.string();
                //std::string str = "C:/Program Files/DAEKYO CNS/VIVEex/MediaServer/plugins/opencv_object_detection_analytics_plugin/yolov4_1_3_416_416_static_50_items.ONNX";

                //std::wstring wide_string = std::wstring(str.begin(), str.end());

                
                //if (with_cuda)
                //{
                //    cuda_options.device_id = 0;
                //    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                //    cuda_options.gpu_mem_limit = static_cast<int>(0.4 * 1024 * 1024 * 1024);  // 4Gb
                //    //cuda_options.gpu_mem_limit = static_cast<int>(SIZE_MAX * 1024 * 1024);
                //    cuda_options.arena_extend_strategy = 1;
                //    cuda_options.do_copy_in_default_stream = 1;
                //    cuda_options.has_user_compute_stream = 1;
                //    cuda_options.default_memory_arena_cfg = nullptr;

                //    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
                //    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                //}
                //
               
                //std::basic_string<ORTCHAR_T> onnxpath = std::basic_string<ORTCHAR_T>(wide_string);
                //onnx_net = new Ort::Session(env, onnxpath.c_str(), sessionOptions);
            }

            std::shared_ptr<Detection> convertRawDetectionToDetection(
                const Mat& rawDetections,
                int detectionIndex,
                const nx::sdk::Uuid trackId)
            {
                enum class OutputIndex
                {
                    classIndex = 1,
                    confidence = 2,
                    xBottomLeft = 3,
                    yBottomLeft = 4,
                    xTopRight = 5,
                    yTopRight = 6,
                };
                static constexpr float confidenceThreshold = 0.5F; //< Chosen arbitrarily.

                const int& i = detectionIndex;
                const float confidence = rawDetections.at<float>(i, (int)OutputIndex::confidence);
                const auto classIndex = (int)(rawDetections.at<float>(i, (int)OutputIndex::classIndex));
                const std::string classLabel = kClasses[(size_t)classIndex];
                const bool confidentDetection = confidence >= confidenceThreshold;
                bool oneOfRequiredClasses = std::find(
                    kClassesToDetect.begin(), kClassesToDetect.end(), classLabel) != kClassesToDetect.end();
                if (confidentDetection && oneOfRequiredClasses)
                {
                    const float xBottomLeft = rawDetections.at<float>(i, (int)OutputIndex::xBottomLeft);
                    const float yBottomLeft = rawDetections.at<float>(i, (int)OutputIndex::yBottomLeft);
                    const float xTopRight = rawDetections.at<float>(i, (int)OutputIndex::xTopRight);
                    const float yTopRight = rawDetections.at<float>(i, (int)OutputIndex::yTopRight);
                    const float width = xTopRight - xBottomLeft;
                    const float height = yTopRight - yBottomLeft;

                    return std::make_shared<Detection>(Detection{
                        /*boundingBox*/ nx::sdk::analytics::Rect(xBottomLeft, yBottomLeft, width, height),
                        classLabel,
                        confidence,
                        trackId
                        });
                }
                return nullptr;
            }


            std::shared_ptr<Detection> convertRawDetectionYoloOnnx(
                cv::Mat frame,
                const Rect boudingbox,
                const int classId,
                const float confidence,
                const nx::sdk::Uuid trackId)
            {
                const std::string classLabel = kClasses[(size_t)classId];
                bool oneOfRequiredClasses = std::find(
                    kClassesToDetect.begin(), kClassesToDetect.end(), classLabel) != kClassesToDetect.end();

                if (oneOfRequiredClasses)
                {
                    const float width = (float)boudingbox.width / (float)frame.cols;
                    const float height = (float)boudingbox.height / (float)frame.rows;
                    const float xTopLeft_ = (float)boudingbox.x / (float)frame.cols;
                    const float yTopLeft_ = (float)boudingbox.y / (float)frame.rows;
                    if (classLabel != "head" && classLabel != "human_fall" && classLabel != "human" && classLabel != "half-fall")
                    {
                        return std::make_shared<Detection>(Detection{
                            /*boundingBox*/ nx::sdk::analytics::Rect(xTopLeft_, yTopLeft_, width, height),
                            classLabel,
                            confidence,
                            trackId
                            });
                    }
                    else
                    {
                        if (confidence >= 0.7F)
                        {
                            return std::make_shared<Detection>(Detection{
                                /*boundingBox*/ nx::sdk::analytics::Rect(xTopLeft_, yTopLeft_, width, height),
                                classLabel,
                                confidence,
                                trackId
                                });
                        }
                    }
                }
                return nullptr;
            }


            //std::shared_ptr<Detection> convertRawDetectionEfficentdetToDetection(
            //    cv::Mat image,
            //    const Mat detections,
            //    int detectionIndex,
            //    const nx::sdk::Uuid trackId)
            //{
            //    enum class OutputIndex
            //    {
            //        x1 = 0,
            //        y1 = 1,
            //        x2 = 2,
            //        y2 = 3,
            //        confidence = 4,
            //        classIndex = 5,
            //    };


            //    const int& i = detectionIndex;

            //    const float confidence = detections.at<float>(i, (int)OutputIndex::confidence);

            //    const auto classIndex = (int)(detections.at<float>(i, (int)OutputIndex::classIndex));

            //    const std::string classLabel = kClasses[(size_t)classIndex];

            //    const bool confidentDetection = confidence >= 0.05;

            //    bool oneOfRequiredClasses = std::find(
            //        kClassesToDetect.begin(), kClassesToDetect.end(), classLabel) != kClassesToDetect.end();

            //    //cv::imwrite("test123_before_bb.png", image);

            //    int in_h = image.rows;
            //    int in_w = image.cols;

            //    if (confidentDetection && oneOfRequiredClasses)
            //    {
            //        float xTopLeft = detections.at<float>(i, (int)OutputIndex::x1);
            //        float yTopLeft = detections.at<float>(i, (int)OutputIndex::y1);
            //        float xBotRight = detections.at<float>(i, (int)OutputIndex::x2);
            //        float yBotRight = detections.at<float>(i, (int)OutputIndex::y2);

            //        float scale;
            //        float offset_w;
            //        float offset_h;

            //        if (in_h > in_w)
            //        {
            //            offset_w = (in_h - in_w) / 2;
            //            offset_h = 0;
            //            scale = in_h / 512.0;
            //        }
            //        else
            //        {
            //            offset_w = 0;
            //            offset_h = (in_w - in_h) / 2;
            //            scale = in_w / 512.0;
            //        }

            //        xTopLeft *= scale;
            //        xTopLeft -= offset_w;

            //        yTopLeft *= scale;
            //        yTopLeft -= offset_h;

            //        xBotRight *= scale;
            //        xBotRight -= offset_w;

            //        yBotRight *= scale;
            //        yBotRight -= offset_h;

            //        const float width = (xBotRight - xTopLeft) / in_w;
            //        const float height = (xBotRight - yTopLeft) / in_h;
            //        const float xTopLeft_ = xTopLeft / in_w;
            //        const float yTopLeft_ = yTopLeft / in_h;


            //        //float width = (xBotRight - xTopLeft) / float(in_w);
            //        //float height = (yBotRight - yTopLeft) / float(in_h);

            //        //cv::rectangle(image, cv::Point(xTopLeft, yTopLeft), cv::Point(xBotRight, yBotRight), cv::Scalar(0, 255, 0), 2);
            //        //cv::rectangle(image, cv::Point(int(xTopLeft), int(yTopLeft)), cv::Point(int(xBotRight), int(yBotRight)), cv::Scalar(0, 0, 255), 2);

            //        //cv::imwrite("test_new.png", image);


            //        return std::make_shared<Detection>(Detection{
            //            /*boundingBox*/ nx::sdk::analytics::Rect(xTopLeft_, yTopLeft_, width, height),
            //            classLabel,
            //            confidence,
            //            trackId
            //            });

            //        //cv::imwrite("test123_with_bb1.png", image);
            //    }

            //    //cv::imwrite("test123_with_bb2.png", image);
            //    return nullptr;
            //}


            //DetectionList ObjectDetector::runImpl(const Frame& frame)
            //{
            //    if (isTerminated())
            //    {
            //        throw ObjectDetectorIsTerminatedError(
            //            "Object detection error: object detector is terminated.");
            //    }

            //    const Mat image = frame.cvMat;

            //    // MobileNet SSD parameters.
            //    static const Size netInputImageSize(300, 300);
            //    static constexpr double scaleFactor = 1.0 / 127.5;
            //    static const Scalar mean(127.5, 127.5, 127.5);
            //    static constexpr int kHeightIndex = 2;
            //    static constexpr int kWidthIndex = 3;

            //    const Mat netInputBlob = blobFromImage(image, scaleFactor, netInputImageSize, mean);

            //    m_net->setInput(netInputBlob);
            //    Mat rawDetections = m_net->forward();
            //    const Mat detections(
            //        /*_rows*/ rawDetections.size[kHeightIndex],
            //        /*_cols*/ rawDetections.size[kWidthIndex],
            //        /*_type*/ CV_32F,
            //        /*_s*/ rawDetections.ptr<float>());

            //    DetectionList result;

            //    for (int i = 0; i < detections.rows; ++i)
            //    {
            //        const std::shared_ptr<Detection> detection = convertRawDetectionToDetection(
            //            /*rawDetections*/ detections,
            //            /*detectionIndex*/ i,
            //            /*trackId*/ m_trackId);
            //        if (detection)
            //        {
            //            result.push_back(detection);
            //            return result;
            //        }
            //    }

            //    return {};
            //}


            DetectionList ObjectDetector::runYoloOnnx_custom(const Frame& frame)
            {
                if (isTerminated())
                {
                    throw ObjectDetectorIsTerminatedError(
                        "Object detection error: object detector is terminated.");
                }
                const Mat image = frame.cvMat;

                blobFromImage(image, blob, 1 / 255.0, yoloInputImageSize, Scalar(0, 0, 0), true, false);

                //Ort::Value intensors = Ort::Value::CreateTensor<float>(memoryInfo, (float*)blob.data, 416 * 416 * 3,
                //    inputDims.data(), inputDims.size());

                inputTensorValues.assign(blob.begin<float>(), blob.end<float>());

                inputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
                    inputDims.size()));

                outputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, outputTensorValues.data(), outputTensorSize,
                    outputDims.data(), outputDims.size()));

                std::vector<const char*> inputNames{ inputName };
                std::vector<const char*> outputNames{ outputName };

                onnx_net->Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);

                const Mat prediction(
                    /*_rows*/ out_shape,
                    /*_cols*/ 6,
                    /*_type*/ CV_32F,
                    /*_s*/ outputTensorValues.data());

                std::vector<int> classIds;
                std::vector<float> confidences;
                std::vector<Rect> boxes;

                for (size_t i = 0; i < prediction.rows; i++)
                {
                    double confidence;

                    confidence = (float)prediction.at<float>(i, 4);

                    //if (confidence > confidence_threshold)
                    //{
                    int left = (int)(prediction.at<float>(i, 0) * image.cols);
                    int top = (int)(prediction.at<float>(i, 1) * image.rows);
                    int right = (int)(prediction.at<float>(i, 2) * image.cols);
                    int bot = (int)(prediction.at<float>(i, 3) * image.rows);

                    int width = right - left;
                    int height = bot - top;

                    int class_id = (int)prediction.at<float>(i, 5);

                    classIds.push_back(class_id);
                    confidences.push_back((float)confidence);

                    boxes.push_back(Rect(left, top, width, height));
                    //}
                }

                std::vector<int> indices;
                NMSBoxes(boxes, confidences, confidence_threshold, nmsThreshold, indices);
                DetectionList result;

                for (int i = 0; i < indices.size(); ++i)  // classIds
                {
                    int idx = indices[i];

                    //Rect box = boxes.at(i);
                    //int clsId = classIds.at(i);
                    //float confidence = confidences.at(i);

                    Rect box = boxes[idx];
                    int clsId = classIds[idx];
                    float confidence = confidences[idx];

                    const std::shared_ptr<Detection> detection = convertRawDetectionYoloOnnx(
                        /*frame*/ image,
                        /*box*/ box,
                        /*clsId*/ clsId,
                        /*confidence*/ confidence,
                        /*trackId*/ m_trackId);

                    if (detection)
                    {
                        result.push_back(detection);
                        return result;
                    }

                }

                return {};
            }

            //DetectionList ObjectDetector::runYoloOnnx(const Frame& frame) {
            //    if (isTerminated())
            //    {
            //        throw ObjectDetectorIsTerminatedError(
            //            "Object detection error: object detector is terminated.");
            //    }

            //    const Mat image = frame.cvMat;

            //    blobFromImage(image, blob, 1 / 255.0, yoloInputImageSize, Scalar(0, 0, 0), true, false);

            //    inputTensorValues.assign(blob.begin<float>(), blob.end<float>());

            //    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            //        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
            //        inputDims.size()));

            //    outputBBTensors.push_back(Ort::Value::CreateTensor<float>(
            //        memoryInfo, outputBBTensorValues.data(), outputBBTensorSize,
            //        outputBBDims.data(), outputBBDims.size()));

            //    outputCfTensors.push_back(Ort::Value::CreateTensor<float>(
            //        memoryInfo, outputCfTensorValues.data(), outputCfTensorSize,
            //        outputCfDims.data(), outputCfDims.size()));

            //    std::vector<const char*> inputNames{ inputName };
            //    std::vector<const char*> outputBBNames{ outputBBName };
            //    std::vector<const char*> outputCfNames{ outputCfName };

            //    onnx_net->Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1, outputBBNames.data(), outputBBTensors.data(), 1);
            //    onnx_net->Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1, outputCfNames.data(), outputCfTensors.data(), 1);


            //    const Mat boudingbox(
            //        /*_rows*/ 10647,
            //        /*_cols*/ 4,
            //        /*_type*/ CV_32F,
            //        /*_s*/ outputBBTensorValues.data());

            //    const Mat confidence(
            //        /*_rows*/ 10647,
            //        /*_cols*/ 3,
            //        /*_type*/ CV_32F,
            //        /*_s*/ outputCfTensorValues.data());

            //    std::vector<int> classIds;
            //    std::vector<float> confidences;
            //    std::vector<Rect> boxes;

            //    for (size_t i = 0; i < boudingbox.rows; i++)
            //    {
            //        Mat scores = confidence.row(i).clone();
            //        Mat bb_per = boudingbox.row(i).clone();

            //        Point classIdPoint;
            //        double confidence;
            //        // Get the value and location of the maximum score
            //        minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            //        if (confidence > 0.1)
            //        {
            //            int left = (int)(boudingbox.at<float>(i, 0) * image.cols);
            //            int top = (int)(boudingbox.at<float>(i, 1) * image.rows);
            //            int right = (int)(boudingbox.at<float>(i, 2) * image.cols);
            //            int bot = (int)(boudingbox.at<float>(i, 3) * image.rows);

            //            int width = right - left;
            //            int height = bot - top;
            //            classIds.push_back(classIdPoint.x);
            //            confidences.push_back((float)confidence);
            //            boxes.push_back(Rect(left, top, width, height));
            //        }
            //    }

            //    std::vector<int> indices;

            //    NMSBoxes(boxes, confidences, 0.1, 0.4, indices);

            //    DetectionList result;

            //    for (int i = 0; i < indices.size(); ++i)
            //    {
            //        int idx = indices[i];

            //        Rect box = boxes[idx];
            //        int clsId = classIds[idx];
            //        float confidence = confidences[idx];

            //        const std::shared_ptr<Detection> detection = convertRawDetectionYoloOnnx(
            //            /*frame*/ image,
            //            /*box*/ box,
            //            /*clsId*/ clsId,
            //            /*confidence*/ confidence,
            //            /*trackId*/ m_trackId);

            //        if (detection)
            //        {
            //            result.push_back(detection);
            //            return result;
            //        }

            //    }

            //    return {};

            //}


            //DetectionList ObjectDetector::runImplCustom(const Frame& frame)
            //{

            //    if (isTerminated())
            //    {
            //        throw ObjectDetectorIsTerminatedError(
            //            "Object detection error: object detector is terminated.");
            //    }

            //    const Mat image = frame.cvMat;

            //    //cv::Mat image = cv::imread("C:/Users/workspace/dataset/fire_smoke/20220627_dau_caleb/dau/20220621_obj/valid/0000000000041.jpg");


            //    // EfficientDet preprocessing

            //    int image_height = image.rows;
            //    int image_width = image.cols;
            //    size_t resized_height;
            //    size_t resized_width;

            //    if (image_height > image_width)
            //    {
            //        resized_height = 512;
            //        resized_width = image_width * (512.0 / image_height);
            //    }
            //    else
            //    {
            //        resized_height = image_height * (512.0 / image_width);
            //        resized_width = 512;
            //    }

            //    Mat img_resize_scale;
            //    cv::resize(image, img_resize_scale, cv::Size(resized_width, resized_height), cv::INTER_LINEAR);

            //    int offset_w = (512 - resized_width) / 2;
            //    int offset_h = (512 - resized_height) / 2;

            //    cv::Mat image_with_pad(512, 512, CV_8UC3);

            //    img_resize_scale.copyTo(image_with_pad(cv::Rect(offset_w, offset_h, img_resize_scale.cols, img_resize_scale.rows)));


            //    //static const Size netInputImageSize(512, 512);
            //    //static constexpr double scaleFactor = 1.0 / 255.0;
            //    cv::Mat feed_img;
            //    cv::dnn::blobFromImage(image_with_pad, feed_img, scaleFactor, netInputImageSize, cv::Scalar(0.4264, 0.4588, 0.4730), false, false, CV_32F);

            //    // Feed to model.

            //    //std::vector<int64_t> inputDims{ 1, 3, 512, 512 };
            //    //std::vector<int64_t> outputDims{ 20, 6 };

            //    //size_t inputTensorSize = vectorProduct(inputDims);
            //    //std::vector<float> inputTensorValues(inputTensorSize);

            //    inputTensorValues.assign(feed_img.begin<float>(),
            //        feed_img.end<float>());

            //    //size_t outputTensorSize = vectorProduct(outputDims);
            //    //assert(("Output tensor size should equal to the label set size.",
            //    //    labels.size() == outputTensorSize));

            //    //std::vector<float> outputTensorValues(outputTensorSize);

            //    //std::vector<Ort::Value> inputTensors;
            //    //std::vector<Ort::Value> outputTensors;

            //    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            //        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
            //    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            //        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
            //        inputDims.size()));
            //    outputTensors.push_back(Ort::Value::CreateTensor<float>(
            //        memoryInfo, outputTensorValues.data(), outputTensorSize,
            //        outputDims.data(), outputDims.size()));

            //    const char* inputName = onnx_net->GetInputName(0, allocator);
            //    const char* outputName = onnx_net->GetOutputName(0, allocator);

            //    std::vector<const char*> inputNames{ inputName };
            //    std::vector<const char*> outputNames{ outputName };

            //    onnx_net->Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);


            //    const Mat detections(
            //        /*_rows*/ 10,
            //        /*_cols*/ 6,
            //        /*_type*/ CV_32F,
            //        /*_s*/ outputTensorValues.data());


            //    DetectionList result;

            //    for (int i = 0; i < detections.rows; ++i)
            //    {
            //        const std::shared_ptr<Detection> detection = convertRawDetectionEfficentdetToDetection(
            //            image,
            //            /*rawDetections*/ detections,
            //            /*detectionIndex*/ i,
            //            /*trackId*/ m_trackId);

            //        if (detection)
            //        {
            //            result.push_back(detection);
            //            return result;
            //        }

            //        //cv::imwrite("test123loop.png", image);
            //    }

            //    //cv::imwrite("test123wtloop.png", image);

            //    return {};
            //}

        } // namespace opencv_object_detection
    } // namespace analytics
} // namespace sample