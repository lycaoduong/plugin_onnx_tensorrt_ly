// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <filesystem>

#include <nx/sdk/analytics/helpers/object_metadata_packet.h>
#include <nx/sdk/analytics/helpers/consuming_device_agent.h>
#include <nx/sdk/helpers/uuid_helper.h>

#include "engine.h"
#include "object_detector.h"
#include "object_tracker.h"

//namespace nx {
//namespace vms_server_plugins {
//namespace analytics {
//namespace sample {
namespace sample {
    namespace analytics {
        namespace opencv_object_detection {

            class DeviceAgent: public nx::sdk::analytics::ConsumingDeviceAgent
            {
            public:
                using MetadataPacketList = std::vector<nx::sdk::Ptr<nx::sdk::analytics::IMetadataPacket>>;

            public:
                DeviceAgent(const nx::sdk::IDeviceInfo* deviceInfo,
                            std::filesystem::path pluginHomeDir, Ort::Session* core_net);
                virtual ~DeviceAgent() override;

            protected:
                virtual std::string manifestString() const override;


                virtual bool pushUncompressedVideoFrame(
                    const nx::sdk::analytics::IUncompressedVideoFrame* videoFrame) override;


                //virtual bool pullMetadataPackets(
                //    std::vector<nx::sdk::analytics::IMetadataPacket*>* metadataPackets) override;

                virtual void doSetNeededMetadataTypes(
                    nx::sdk::Result<void>* outValue,
                    const nx::sdk::analytics::IMetadataTypes* neededMetadataTypes) override;

            private:

                void reinitializeObjectTrackerOnFrameSizeChanges(const Frame& frame);
                //nx::sdk::Ptr<nx::sdk::analytics::IMetadataPacket> generateEventMetadataPacket();
                /* Ly Cao Duong */
                nx::sdk::Ptr<nx::sdk::analytics::ObjectMetadataPacket> detectionsToObjectMetadataPacket(
                    const DetectionList& detections,
                    int64_t timestampUs);

                MetadataPacketList eventsToEventMetadataPacketList(
                    const EventList& events,
                    int64_t timestampUs);

                MetadataPacketList processFrame(
                    const nx::sdk::analytics::IUncompressedVideoFrame* videoFrame);

                /**/

            private:
                //const std::string kCatObjectType = "samle.opencv_object_detection.cat";
                //const std::string kDogObjectType = "samle.opencv_object_detection.dog";
                const std::string kHelmetObjectType = "nx.base.helmet";
                const std::string kFireObjectType = "nx.base.fire";
                const std::string kSmokeObjectType = "nx.base.smoke";
                const std::string kHumanFallObjectType = "nx.base.human_fall";
                const std::string kHeadtObjectType = "nx.base.head";;
                const std::string kNonHelmetObjectType = "nx.base.non_helmet";
                const std::string kHumanObjectType = "nx.base.human";
                const std::string kHalfFallObjectType = "nx.base.half-fall";

                //const std::string kNewTrackEventType = "nx.sample.newTrack";

                const std::string kDetectionEventType = "sample.opencv_object_detection.detection";
                const std::string kDetectionEventCaptionSuffix = " detected";
                const std::string kDetectionEventDescriptionSuffix = " detected";

                const std::string kProlongedDetectionEventType =
                    "sample.opencv_object_detection.prolongedDetection";

                /** Lenght of the the track (in frames). The value was chosen arbitrarily. */
                //static constexpr int kTrackFrameCount = 256;

                /** Should work on modern PCs. */
                static constexpr int kDetectionFramePeriod = 2;

            private:
                //nx::sdk::Uuid person_trackId = nx::sdk::UuidHelper::randomUuid();
                //nx::sdk::Uuid cat_trackId = nx::sdk::UuidHelper::randomUuid();
                //nx::sdk::Uuid dog_trackId = nx::sdk::UuidHelper::randomUuid();

                bool m_terminated = false;
                bool m_terminatedPrevious = false;
                const std::unique_ptr<ObjectDetector> m_objectDetector;
                std::unique_ptr<ObjectTracker> m_objectTracker;

                //nx::sdk::Uuid m_trackId = nx::sdk::UuidHelper::randomUuid();

                int m_frameIndex = 0; /**< Used for generating the detection in the right place. */
                //int m_trackIndex = 0; /**< Used in the description of the events. */

                /** Used for binding object and event metadata to the particular video frame. */
                //int64_t m_lastVideoFrameTimestampUs = 0;

                /** Used for checking whether frame size changed and reinitializing the tracker. */
                int m_previousFrameWidth = 0;
                int m_previousFrameHeight = 0;

                //int imgCount = 1;
            };

        } // namespace opencv_object_detection
    } // namespace analytics
} // namespace sample

//} // namespace sample
//} // namespace analytics
//} // namespace vms_server_plugins
//} // namespace nx
