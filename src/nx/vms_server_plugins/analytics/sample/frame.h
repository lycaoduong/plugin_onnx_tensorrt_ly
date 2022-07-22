#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <nx/sdk/analytics/i_uncompressed_video_frame.h>
//#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>

//#include <nx/kit/debug.h>
//#include <nx/kit/utils.h>
//#include <fstream>

namespace sample {
    namespace analytics {
        namespace opencv_object_detection {
            struct Frame
            {
                const int width;
                const int height;
                const int64_t timestampUs;
                const int64_t index;
                const cv::Mat cvMat;
            public:
                Frame(const nx::sdk::analytics::IUncompressedVideoFrame* frame, int64_t index) :
                    width(frame->width()),
                    height(frame->height()),
                    timestampUs(frame->timestampUs()),
                    index(index),
                    cvMat({
                    /*_rows*/ height,
                    /*_cols*/ width,
                    /*_type*/ CV_8UC3, //< BGR color space (default for OpenCV)
                    /*_data*/ (void*)frame->data(0),
                    /*_step*/ (size_t)frame->lineSize(0),
                        })
                {
                }

                //Frame(const nx::sdk::analytics::IUncompressedVideoFrame* frame, int64_t index) :
                //    width(frame->width()),
                //    height(frame->height()),
                //    timestampUs(frame->timestampUs()),
                //    index(index),
                //    cvMat(ndk_yuv_to_rgb_image(frame))
                //{
                //}
            private:
                cv::Mat ndk_yuv_to_rgb_image(const nx::sdk::analytics::IUncompressedVideoFrame* videoFrame)
                {
                    int frame_width = videoFrame->width();
                    int frame_height = videoFrame->height();

                    //uint8_t* yPixel = nullptr;
                    //uint8_t* uPixel = nullptr;
                    //uint8_t* vPixel = nullptr;

                    //NX_PRINT << __func__ << " plane count= " << videoFrame->planeCount();

                    char* yPixel = (char*)videoFrame->data(0);
                    char* uPixel = (char*)videoFrame->data(1);
                    char* vPixel = (char*)videoFrame->data(2);

                    //int32_t yLen = 0;
                    //int32_t uLen = 0;
                    //int32_t vLen = 0;

                    int32_t yLen = (size_t)videoFrame->dataSize(0);
                    int32_t uLen = (size_t)videoFrame->dataSize(1);
                    int32_t vLen = (size_t)videoFrame->dataSize(2);

                    cv::Mat _yuv_rgb_img;

                    //AImage_getPlaneData(yuv_image, 0, &yPixel, &yLen);
                    //AImage_getPlaneData(yuv_image, 1, &uPixel, &uLen);
                    //AImage_getPlaneData(yuv_image, 2, &vPixel, &vLen);

                    uint8_t* data = new uint8_t[yLen + vLen + uLen];

                    memcpy(data, yPixel, yLen);
                    memcpy(data + yLen, vPixel, vLen);
                    memcpy(data + yLen + vLen, uPixel, uLen);

                    cv::Mat mYUV = cv::Mat(frame_height * 1.5, frame_width, CV_8UC1, data);

                    cv::cvtColor(mYUV, _yuv_rgb_img, cv::COLOR_YUV2BGR_I420);                    
                    //cv::cvtColor(mYUV, _yuv_rgb_img, cv::COLOR_YCrCb2RGB);

                    //cv::cuda::GpuMat bgr_gpu, yuv_gpu;

                    //yuv_gpu.upload(mYUV);
                    //cv::cuda::cvtColor(yuv_gpu, bgr_gpu, cv::COLOR_YUV2BGR_I420);

                    //bgr_gpu.download(_yuv_rgb_img);

                    return _yuv_rgb_img;
                }

            };
                
        } // namespace opencv_object_detection
    } // namespace analytics
} // namespace sample
